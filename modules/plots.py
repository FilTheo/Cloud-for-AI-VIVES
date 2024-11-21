import boto3
from io import BytesIO
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from logs.logging_config import setup_logger
import plotly.io as pio

logger = setup_logger(__name__, "plots.log")

def make_plots(evaluation_df, config_manager, full_data):
    """
    Generate interactive Plotly plots based on evaluation data and save as HTML.

    Args:
        evaluation_df (pd.DataFrame): DataFrame containing evaluation metrics.
        config_manager (ConfigManager): The configuration manager object.
        full_data (pd.DataFrame, optional): Full dataset for additional plots.

    Returns:
        dict: Dictionary of plot names and their HTML representations.
    """
    plot_data = {}

    # convert date to timestamp
    evaluation_df['date'] = pd.to_datetime(evaluation_df['date'])

    try:
        # Plot 1: Item Model Comparison
        fig1 = plot_item_model_comparison(evaluation_df, metric='mae', max_items=40)
        plot_data['Worse Predictions'] = pio.to_html(fig1, full_html=False)
        
        # Plot 2: Metrics Over Time
        fig2 = plot_metrics(evaluation_df, mode='mean')
        plot_data['Performance over time'] = pio.to_html(fig2, full_html=False)
        
        # Plot 3: Evaluation Heatmap
        fig3 = plot_eval_heatmap(evaluation_df, metric='rmsse', overall=True)
        plot_data['Total Items'] = pio.to_html(fig3, full_html=False)

        # New Plot: Confirmed MAE
        fig4 = plot_confirmed_vs_true(evaluation_df)
        if fig4 is not None:
            plot_data['Confirmed MAE'] = pio.to_html(fig4, full_html=False)

        
        fig5 = plot_seasonal_pattern(full_data)
        plot_data['Seasonal Pattern'] = pio.to_html(fig5, full_html=False)

        # Save plots to S3
        s3_config = config_manager.get_s3_config()
        s3 = boto3.client('s3')
        
        for plot_name, plot_html in plot_data.items():
            s3.put_object(
                Bucket=s3_config['bucket_name'],
                Key=f"{s3_config['plots_path']}/{plot_name}.html",
                Body=plot_html,
                ContentType='text/html'
            )

        logger.info(f"Generated and saved interactive plots as HTML: {list(plot_data.keys())}")
    except Exception as e:
        logger.error(f"Error generating interactive plots: {str(e)}", exc_info=True)

    return plot_data

def plot_eval_heatmap(eval_df, metric='mae', overall=False):
    # Step 1: Prepare the data
    if overall:
        # Calculate mean across all dates for each item
        plot_df = eval_df.groupby('unique_id')[metric].mean().reset_index()
    else:
        # Get the latest date for each item
        latest_date = eval_df['date'].max()
        plot_df = eval_df[eval_df['date'] == latest_date][['unique_id', metric]]
    
    # Step 2: Reshape the data for heatmap
    plot_df = plot_df.set_index('unique_id')
    plot_df = plot_df.sort_values(by=metric, ascending=False)
    
    # Calculate number of rows and columns for the grid
    n_items = len(plot_df)
    n_cols = int(np.ceil(np.sqrt(n_items)))
    n_rows = int(np.ceil(n_items / n_cols))
    
    # Reshape the data into a 2D array
    heatmap_data = np.full((n_rows, n_cols), np.nan)
    item_names = np.full((n_rows, n_cols), '', dtype=object)
    for i, (index, value) in enumerate(plot_df.iterrows()):
        row = i // n_cols
        col = i % n_cols
        heatmap_data[row, col] = value[metric]
        item_names[row, col] = index
    
    # Step 3: Create the interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        text=item_names,
        hoverinfo='text+z',
        colorscale='RdYlGn_r',
        zmid=np.nanmedian(heatmap_data),
        showscale=True
    ))
    
    # Step 4: Add grid lines
    fig.update_layout(
        title=f"{metric.upper()} Heatmap ({'Overall' if overall else 'Latest Date'})",
        xaxis=dict(showticklabels=False, ticks='', showgrid=True, gridcolor='black', gridwidth=1),
        yaxis=dict(showticklabels=False, ticks='', showgrid=True, gridcolor='black', gridwidth=1),
        coloraxis_colorbar=dict(title=metric.upper())
    )
    
    return fig

def plot_item_model_comparison(eval_df, metric='mae', max_items=40):
    # Step 1: Prepare the data
    latest_date = eval_df['date'].max()
    latest_df = eval_df[eval_df['date'] == latest_date]
    
    # Filter for AI model and sort by the metric
    ai_df = latest_df[latest_df['Model'] == 'SeasonXpert'].sort_values(by=metric, ascending=False)
    
    # Get the top items based on AI performance
    top_items = ai_df['unique_id'].head(max_items).tolist()
    
    # Filter data for top items and both models
    plot_df = latest_df[latest_df['unique_id'].isin(top_items)]
    
    # Create dataframes for each model, ordered by AI scores
    ai_plot_df = plot_df[plot_df['Model'] == 'SeasonXpert'].sort_values(by=metric, ascending=True)
    lastweek_plot_df = plot_df[plot_df['Model'] == 'SeasonalNaive'].set_index('unique_id').loc[ai_plot_df['unique_id']].reset_index()
    
    # Step 2: Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("SeasonXpert", "Benchmark Model"))
    
    # Step 3: Add traces for each model
    fig.add_trace(
        go.Bar(
            name='SeasonXpert',
            x=ai_plot_df['unique_id'],
            y=ai_plot_df[metric],
            text=ai_plot_df[metric].round(2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                          f'{metric.upper()} (SeasonXpert): %{{y:.2f}}<br>' +
                          f'{metric.upper()} (SeasonalNaive): %{{customdata:.2f}}',
            customdata=lastweek_plot_df[metric],
            marker=dict(color='rgba(58, 71, 80, 0.8)', line=dict(color='rgba(58, 71, 80, 1.0)', width=1))
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='SeasonalNaive',
            x=lastweek_plot_df['unique_id'],
            y=lastweek_plot_df[metric],
            text=lastweek_plot_df[metric].round(2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                          f'{metric.upper()} (SeasonalNaive): %{{y:.2f}}<br>' +
                          f'{metric.upper()} (SeasonXpert): %{{customdata:.2f}}',
            customdata=ai_plot_df[metric],
            marker=dict(color='rgba(246, 78, 139, 0.8)', line=dict(color='rgba(246, 78, 139, 1.0)', width=1))
        ),
        row=2, col=1
    )
    
    # Step 4: Update layout
    y_max = max(ai_plot_df[metric].max(), lastweek_plot_df[metric].max())
    y_min = min(ai_plot_df[metric].min(), lastweek_plot_df[metric].min())
    y_range = [max(0, y_min - 0.1 * (y_max - y_min)), y_max * 1.1]
    
    fig.update_layout(
        title=f"{metric.upper()} Comparison by Item and Model (Latest Date: {latest_date})",
        height=800,
        width=max(1200, len(top_items) * 25),  # Adjust width based on number of items
        showlegend=False,
        bargap=0.05,  # Reduce gap between bars
        bargroupgap=0.05  # Reduce gap between bar groups
    )
    
    # Update y-axes to have the same range
    fig.update_yaxes(title_text=metric.upper(), row=1, col=1, range=y_range)
    fig.update_yaxes(title_text=metric.upper(), row=2, col=1, range=y_range)
    
    # Hide x-axis labels
    fig.update_xaxes(showticklabels=False)
    
    return fig

def plot_metrics(eval_df, mode='mean'):
    eval_df['abs_bias'] = eval_df['bias'].abs()
    
    if mode == 'mean':
        # Calculate mean metrics per date and Model
        mean_df = eval_df.groupby(['date', 'Model']).agg({'mae': 'mean', 'abs_bias': 'mean'}).reset_index()
        
        fig = go.Figure()
        
        for model in mean_df['Model'].unique():
            model_df = mean_df[mean_df['Model'] == model]
            
            # Plot RMSSE
            fig.add_trace(go.Scatter(
                x=model_df['date'], y=model_df['mae'], mode='lines', name=f'MAE ({model})',
                hovertemplate='Date: %{x}<br>RMSSE: %{y:.2f}<br>Model: ' + model
            ))
            
            # Plot Absolute Bias on secondary y-axis
            fig.add_trace(go.Scatter(
                x=model_df['date'], y=model_df['abs_bias'], mode='lines', name=f'Absolute Bias ({model})',
                line=dict(dash='dash'),
                hovertemplate='Date: %{x}<br>Absolute Bias: %{y:.2f}<br>Model: ' + model,
                yaxis='y2'
            ))
        
        # Update layout for dual y-axes
        fig.update_layout(
            title=f'Metrics Over Time ({mode.capitalize()} View)',
            xaxis_title='Date',
            yaxis=dict(title='MAE'),
            yaxis2=dict(title='Absolute Bias', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
    elif mode == 'full':
        fig = go.Figure()
        
        for model in eval_df['Model'].unique():
            model_df = eval_df[eval_df['Model'] == model]
            for item in model_df['unique_id'].unique():
                item_df = model_df[model_df['unique_id'] == item]
                fig.add_trace(go.Scatter(
                    x=item_df['date'], y=item_df['mae'], mode='lines', line=dict(color='gray', width=1),
                    opacity=0.1, showlegend=False, hoverinfo='skip'
                ))
            
            # Plot mean RMSSE for each model
            mean_df = model_df.groupby('date')['mae'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=mean_df['date'], y=mean_df['mae'], mode='lines', name=f'Mean MAE ({model})',
                line=dict(width=2),
                hovertemplate='Date: %{x}<br>Mean RMSSE: %{y:.2f}<br>Model: ' + model
            ))
        
        fig.update_layout(
            title=f'Metrics Over Time ({mode.capitalize()} View)',
            xaxis_title='Date',
            yaxis_title='MAE',
            legend=dict(x=0, y=1.1, orientation='h')
        )
    
    return fig

def plot_confirmed_vs_true(eval_df):
    # Check if 'Confirmed' model exists in the data
    if 'Confirmed' not in eval_df['Model'].unique():
        return None

    # Filter data for Confirmed model
    confirmed_df = eval_df[eval_df['Model'] == 'Confirmed'].copy()

    # Sort by MAE and get top 40 items
    top_items = confirmed_df.sort_values('mae', ascending=False).head(40)

    # Create the bar plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_items['unique_id'],
        y=top_items['mae'],
        name='MAE (Confirmed)',
        marker_color=top_items['mae'].apply(lambda x: 'red' if x > 0.5 else 'green'),
        text=top_items['mae'].round(2),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>' +
                      'MAE: %{y:.2f}<br>' +
                      'RMSSE: %{customdata[0]:.2f}<br>' +
                      'Bias: %{customdata[1]:.2f}',
        customdata=top_items[['rmsse', 'bias']]
    ))

    fig.update_layout(
        title='Top 40 Items: MAE for Confirmed Values',
        xaxis_title='Item ID',
        yaxis_title='MAE',
        height=600,
        width=1200,
        bargap=0.2,
        xaxis_tickangle=-45
    )

    return fig


def plot_seasonal_pattern(full_data):
    """
    Create an interactive seasonal pattern plot showing daily sales across recent weeks.
    
    Args:
        full_data (pd.DataFrame): The full dataset containing date and sales data
        
    Returns:
        go.Figure: Plotly figure object containing the seasonal pattern plot
    """

    # Get last 4 weeks of data
    last_date = full_data['date'].max()
    last_monday = last_date - pd.Timedelta(days=last_date.weekday())
    four_weeks_ago = last_monday - pd.Timedelta(weeks=3)
    full_data = full_data[full_data['date'] >= four_weeks_ago]
    
    # Add day name and week information
    full_data['day_name'] = full_data['date'].dt.day_name()
    last_week = full_data['date'].dt.isocalendar().week.max()
    full_data['Week'] = full_data['date'].dt.isocalendar().week.apply(lambda x: last_week - x + 1)
    
    # Map week numbers to descriptive names
    week_names = {
        1: 'Current Week',
        2: 'Last Week', 
        3: 'Two Weeks Ago',
        4: 'Three Weeks Ago'
    }
    
    # Calculate daily sums
    temp_sums = full_data.groupby(['Week', 'day_name'])['y'].sum().reset_index()
    temp_sums['Week'] = temp_sums['Week'].map(week_names)
    
    # Create the interactive plot
    fig = go.Figure()
    
    # Add bars for each week
    for week in week_names.values():
        week_data = temp_sums[temp_sums['Week'] == week]
        fig.add_trace(go.Bar(
            name=week,
            x=week_data['day_name'],
            y=week_data['y'],
            hovertemplate='Day: %{x}<br>' +
                         'Sales: %{y:,.0f}<br>' +
                         'Week: ' + week
        ))
    
    # Update layout
    fig.update_layout(
        title='Total Daily Sales Comparison Across Last 4 Weeks',
        xaxis_title='Day of Week',
        yaxis_title='Total Sales Volume',
        barmode='group',
        plot_bgcolor='rgb(237, 237, 237)',
        width=1200,
        height=600,
        showlegend=True,
        legend=dict(
            title='Week',
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15
        )
    )
    
    # Add gridlines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white')
    
    return fig