from logs.logging_config import setup_logger
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify,
    session
)
from functools import wraps
from datetime import datetime
import json
import re
import os
from modules.data_extractor import handle_file_upload
from modules.client_config_manager import ConfigManager
from modules.utils.configurations.m5 import DEMO_AWS_LOAD_CONFIGURATIONS
import pandas as pd
import io
from modules.pipeline import run_pipeline, get_latest_files
from modules.utils.user_configurations import PASSWORDS
from tenacity import retry, stop_after_attempt, wait_exponential
import boto3
from modules.data_extractor import Extractor
from dotenv import load_dotenv

load_dotenv()

GENERAL_PASSWORD = "vives_tetra_product"

# Add this near the top of the file, after the imports
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logger
logger = setup_logger(__name__, 'logs/app.log')

# Add this near the top of the file, after the imports
logger.info(f"AWS_ACCESS_KEY_ID set: {'AWS_ACCESS_KEY_ID' in os.environ}")
logger.info(f"AWS_SECRET_ACCESS_KEY set: {'AWS_SECRET_ACCESS_KEY' in os.environ}")
logger.info(f"AWS_DEFAULT_REGION set: {'AWS_DEFAULT_REGION' in os.environ}")

# Try to create a boto3 client and log the result
try:
    s3 = boto3.client('s3')
    logger.info("Successfully created boto3 S3 client")
except Exception as e:
    logger.error(f"Failed to create boto3 S3 client: {str(e)}")

# Add this function to test AWS credentials
def test_aws_credentials():
    try:
        sts = boto3.client('sts')
        response = sts.get_caller_identity()
        logger.info(f"AWS credentials are valid. Account ID: {response['Account']}")
    except Exception as e:
        logger.error(f"AWS credentials are invalid or not set: {str(e)}")

# Call the function to test credentials
test_aws_credentials()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Initialize ConfigManager as None
config_manager = None

# Simplified login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'client' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Simplified login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        general_password = request.form['general_password']
        client_password = request.form['password']
        
        if general_password != GENERAL_PASSWORD:
            flash('Invalid general password')
            return render_template('login.html')
        
        client_folder = PASSWORDS.get(client_password)
        if client_folder:
            session['client'] = client_folder
            return redirect(url_for('index'))
        flash('Invalid client password')
    return render_template('login.html')

# Simplified logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/")
@login_required
def index():
    client = session['client']
    config_manager = ConfigManager(os.getenv("BUCKET_NAME"), client)
    s3_config = config_manager.get_s3_config()
    
    # Load predictions
    predictions = None
    try:
        predictions_data = config_manager.s3.get_object(
            Bucket=s3_config['bucket_name'],
            Key=f"{s3_config['model_path']}/latest_predictions.json"
        )['Body'].read().decode('utf-8')
        predictions = json.loads(predictions_data)
        logger.info(f"Predictions loaded successfully")
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}", exc_info=True)
        flash(f"Error loading predictions: {str(e)}", "error")

    # Process predictions for the template
    processed_predictions = {}
    if predictions:
        for key, values in predictions.items():
            processed_predictions[key] = {}
            for item in values:
                date = item['date']
                model = item['Model']
                if date not in processed_predictions[key]:
                    processed_predictions[key][date] = {
                        'SeasonXpert': None, 
                        'Vision': None, 
                        'Confirmed': None,  # Change this to None
                        'True': None,  # Change this to None
                        'Color': item['Color']
                    }
                processed_predictions[key][date][model] = item['y'] if pd.notna(item['y']) else None
                
                # Set True value only if it's not None or NaN
                if pd.notna(item['True']):
                    processed_predictions[key][date]['True'] = item['True']

    # Load interactive plots from S3
    plots = {}
    try:
        response = config_manager.s3.list_objects_v2(
            Bucket=s3_config['bucket_name'],
            Prefix=f"{s3_config['plots_path']}/"
        )
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.html'):
                plot_name = obj['Key'].split('/')[-1].split('.')[0]
                plot_data = config_manager.s3.get_object(
                    Bucket=s3_config['bucket_name'],
                    Key=obj['Key']
                )['Body'].read().decode('utf-8')
                plots[plot_name] = plot_data
        logger.info(f"Interactive plots loaded: {list(plots.keys())}")
    except Exception as e:
        logger.error(f"Error fetching interactive plots from S3: {str(e)}", exc_info=True)
        flash(f"Error fetching interactive plots: {str(e)}", "error")

    return render_template(
        "dashboard.html", predictions=processed_predictions, plots=plots, client=client
    )


@app.route("/upload_file", methods=["POST"])
@login_required
def upload_file():
    logger.info("File upload attempt")
    if "file" not in request.files:
        logger.warning("No file part in the request")
        return jsonify({"success": False, "message": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "message": "No selected file"})

    if file:
        # Check if the file name matches the required format
        if not re.match(r"^\d{6}\.(csv|xlsx)$", file.filename):
            return jsonify(
                {
                    "success": False,
                    "message": "File name should be in the format DDMMYY.csv or DDMMYY.xlsx",
                }
            )

        config_manager = ConfigManager(os.getenv("BUCKET_NAME"), session['client'])
        s3_config = config_manager.get_s3_config()
        success = handle_file_upload(file, s3_config['bucket_name'], s3_config['data_path'])
        if success:
            logger.info(f"File {file.filename} uploaded successfully")
            return jsonify({"success": True, "message": "File uploaded successfully"})
        else:
            logger.error(f"File {file.filename} upload failed")
            return jsonify({"success": False, "message": "File upload failed"})


@app.route("/run_forecast", methods=["POST"])
@login_required
def run_forecast():
    logger.info("Running forecast")
    try:
        config_manager = ConfigManager(os.getenv("BUCKET_NAME"), session['client'])
        run_pipeline(config_manager)
        logger.info("New forecasts generated successfully")
        return jsonify({'success': True, 'message': "New forecasts generated successfully!"})
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}", exc_info=True)
        error_message = f"Error generating forecasts: {str(e)}"
        if "NoSuchKey" in str(e):
            error_message += " (File not found in S3)"
        return jsonify({'success': False, 'message': error_message})

@app.route("/download_orders", methods=["POST"])
@login_required
def download_orders():
    logger.info("Downloading orders")
    
    config_manager = ConfigManager(os.getenv("BUCKET_NAME"), session['client'])
    s3_config = config_manager.get_s3_config()
    s3 = boto3.client('s3')
    
    try:
        response = s3.get_object(Bucket=s3_config['bucket_name'], Key=f"{s3_config['model_path']}/latest_predictions.json")
        predictions_json = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        logger.error(f"Error loading predictions JSON: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': "Error loading predictions data"})

    # Process predictions
    processed_predictions = {}
    for key, values in predictions_json.items():
        processed_predictions[key] = {}
        for item in values:
            date = item['date']
            model = item['Model']
            if date not in processed_predictions[key]:
                processed_predictions[key][date] = {
                    'SeasonXpert': '', 
                    'SeasonalNaive': '', 
                    'Confirmed': '', 
                    'True': ''
                }
            processed_predictions[key][date][model] = item['y'] if pd.notna(item['y']) else ''
            if pd.notna(item['True']):
                processed_predictions[key][date]['True'] = item['True']

    # Extract unique dates and sort them
    all_dates = sorted(set(date for preds in processed_predictions.values() for date in preds.keys()))

    # Create DataFrame
    data = []
    for unique_id, predictions in processed_predictions.items():
        row = {'Unique ID': unique_id}
        for date in all_dates:
            if date in predictions:
                row[f"{date}_SeasonXpert"] = predictions[date]['SeasonXpert']
                row[f"{date}_SeasonalNaive"] = predictions[date]['SeasonalNaive']
                row[f"{date}_Confirmed"] = predictions[date]['Confirmed']
                row[f"{date}_True"] = predictions[date]['True']
            else:
                row[f"{date}_SeasonXpert"] = ''
                row[f"{date}_SeasonalNaive"] = ''
                row[f"{date}_Confirmed"] = ''
                row[f"{date}_True"] = ''
        data.append(row)

    df = pd.DataFrame(data)

    # Create an Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Orders", index=False, startrow=1)

        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Orders']

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'align': 'center',
            'border': 1
        })
        date_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'border': 1
        })

        # Write the column headers
        worksheet.write(0, 0, 'Unique ID', header_format)
        for col_num, date in enumerate(all_dates):
            worksheet.merge_range(0, col_num*4 + 1, 0, col_num*4 + 4, date, date_format)
            worksheet.write(1, col_num*4 + 1, 'SeasonXpert', header_format)
            worksheet.write(1, col_num*4 + 2, 'SeasonalNaive', header_format)
            worksheet.write(1, col_num*4 + 3, 'Confirmed', header_format)
            worksheet.write(1, col_num*4 + 4, 'True', header_format)

        # Set column widths
        worksheet.set_column(0, 0, 15)  # Width of Unique ID column
        worksheet.set_column(1, len(all_dates)*4, 12)  # Width of data columns

        # Add borders
        border_format = workbook.add_format({'border': 1})
        worksheet.conditional_format(0, 0, len(data) + 1, len(all_dates)*4,
                                     {'type': 'no_blanks', 'format': border_format})

    output.seek(0)

    # Send the file for download
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="orders.xlsx",
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_pipeline_with_retry(config_manager):
    return run_pipeline(config_manager)

@app.route("/update_data", methods=["POST"])
@login_required
def update_data():
    logger.info("Updating data")
    try:
        config_manager = ConfigManager(os.getenv("BUCKET_NAME"), session['client'])
        s3_config = config_manager.get_s3_config()
        
        # Get the latest files
        latest_file, latest_main_file, _, _, _ = get_latest_files(config_manager)

        extractor = Extractor(
            log_in_aws=True,
            bucket_path=s3_config['data_path'],
            bucket_name=s3_config['bucket_name'],
            original_file_name=latest_main_file,
            new_file_name=latest_file,
            client_name=session['client']
        )
        
        if latest_file is None:
            return jsonify({'success': False, 'message': "No new data to merge."})
        
        temp_original, temp_new = extractor.extract_data()
        
        extractor.merge_new_data()
        logger.info("Data updated successfully")
        return jsonify({'success': True, 'message': "Data updated successfully!"})
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': f"Error updating data: {str(e)}"})

if __name__ == "__main__":
    logger.info("Starting the Flask application")
    app.run(debug=True, host="0.0.0.0", port=5000)
