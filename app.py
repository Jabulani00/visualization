from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import os
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # For flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# File extensions we'll allow
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

def read_file(file):
    """Read CSV or Excel file into pandas DataFrame"""
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    print(f"Reading file: {filename}, File exists: {os.path.exists(file_path)}")
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            print("Reading .xlsx file with openpyxl")
            df = pd.read_excel(file_path, engine='openpyxl')
        elif filename.endswith('.xls'):
            print("Reading .xls file with xlrd")
            df = pd.read_excel(file_path, engine='xlrd')
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        print(f"Successfully read file. Shape: {df.shape}")
        # Clean up temp file
        os.remove(file_path)
        return df
    except Exception as e:
        # Don't remove the file on error so we can investigate
        print(f"Error reading file: {str(e)}")
        raise

def generate_summary_stats(df):
    """Generate summary statistics for DataFrame"""
    # Basic info
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"Generating summary stats for {len(numeric_cols)} numeric columns")
    
    if len(numeric_cols) > 0:
        summary = df[numeric_cols].describe().T
        summary['missing'] = df[numeric_cols].isnull().sum()
        summary['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df)) * 100
        summary = summary.round(2)
        return summary.to_html(classes="table table-striped table-hover")
    else:
        return "<p>No numeric columns found for summary statistics.</p>"

def detect_outliers(df):
    """Detect outliers in numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"Detecting outliers for {len(numeric_cols)} numeric columns")
    
    if len(numeric_cols) == 0:
        return "<p>No numeric columns found for outlier detection.</p>"
    
    # Create subplots for boxplots
    fig = make_subplots(rows=len(numeric_cols), cols=1, 
                        subplot_titles=numeric_cols,
                        vertical_spacing=0.05)
    
    for i, col in enumerate(numeric_cols, 1):
        fig.add_trace(go.Box(y=df[col], name=col), row=i, col=1)
    
    fig.update_layout(height=300 * len(numeric_cols), showlegend=False, 
                     title_text="Outlier Analysis with Box Plots")
    
    # Use include_plotlyjs='cdn' to ensure Plotly is loaded
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def generate_correlation_plot(df):
    """Generate correlation heatmap for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    print(f"Generating correlation plot for {numeric_df.shape[1]} numeric columns")
    
    if numeric_df.shape[1] < 2:
        return "<p>Not enough numeric columns for correlation analysis.</p>"
    
    corr = numeric_df.corr()
    
    # Create correlation heatmap
    fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1))
    
    fig.update_layout(
        title='Correlation Heatmap',
        height=600,
        width=800)
    
    # Use include_plotlyjs='cdn' to ensure Plotly is loaded
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def compare_datasets(df1, df2):
    """Compare two datasets and highlight differences"""
    # Check for common columns
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    print(f"Common columns found for comparison: {len(common_cols)}")
    
    if not common_cols:
        return "<p>No common columns found to compare datasets.</p>"
    
    # For numeric columns, compare distributions
    numeric_cols = [col for col in common_cols if col in df1.select_dtypes(include=[np.number]).columns
                   and col in df2.select_dtypes(include=[np.number]).columns]
    
    comparison_html = "<h3>Common Columns: " + ", ".join(common_cols) + "</h3>"
    
    if numeric_cols:
        fig = make_subplots(rows=len(numeric_cols), cols=1, 
                           subplot_titles=numeric_cols,
                           vertical_spacing=0.1)
        
        for i, col in enumerate(numeric_cols, 1):
            fig.add_trace(go.Histogram(x=df1[col], name='File 1', opacity=0.7), row=i, col=1)
            fig.add_trace(go.Histogram(x=df2[col], name='File 2', opacity=0.7), row=i, col=1)
        
        fig.update_layout(height=400 * len(numeric_cols), barmode='overlay',
                        title_text="Distribution Comparison for Numeric Columns")
        
        # Use include_plotlyjs='cdn' to ensure Plotly is loaded
        comparison_html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Generate statistics comparison table
    stats_comparison = {
        'Number of Rows': [len(df1), len(df2)],
        'Number of Columns': [len(df1.columns), len(df2.columns)],
        'Common Columns': [len(common_cols), len(common_cols)],
        'Missing Values': [df1[common_cols].isnull().sum().sum(), df2[common_cols].isnull().sum().sum()]
    }
    
    stats_df = pd.DataFrame(stats_comparison, index=['File 1', 'File 2'])
    comparison_html += "<h3>Dataset Statistics Comparison</h3>"
    comparison_html += stats_df.to_html(classes="table table-striped")
    
    return comparison_html

def generate_visualizations(df):
    """Generate various visualizations based on data"""
    visualizations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"Number of numeric columns found for visualization: {len(numeric_cols)}")
    
    if len(numeric_cols) < 1:
        print("No numeric columns found for visualization")
        return visualizations
    
    # Track if we've included Plotly.js yet
    plotlyjs_included = False
    
    # 1. Distribution plot for first numeric column
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        print(f"Creating histogram for {col}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[col], nbinsx=30))
        fig.update_layout(title=f"Distribution of {col}", 
                         xaxis_title=col, 
                         yaxis_title="Count")
        
        # First visualization includes the Plotly.js CDN
        visualizations.append({
            'title': f"Distribution of {col}",
            'plot': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'description': f"Histogram showing the frequency distribution of {col} values."
        })
        plotlyjs_included = True
    
    # 2. Scatter plot for first two numeric columns if available
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        print(f"Creating scatter plot for {col1} vs {col2}")
        fig = px.scatter(df, x=col1, y=col2)
        fig.update_layout(title=f"Relationship between {col1} and {col2}")
        
        # For subsequent visualizations, don't include Plotly.js again
        include_js = 'cdn' if not plotlyjs_included else False
        visualizations.append({
            'title': f"Scatter Plot: {col1} vs {col2}",
            'plot': fig.to_html(full_html=False, include_plotlyjs=include_js),
            'description': f"Scatter plot showing the relationship between {col1} and {col2}."
        })
        plotlyjs_included = True
    
    # 3. Bar chart for first categorical column if available
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        col = categorical_cols[0]
        print(f"Creating bar chart for categorical column: {col}")
        value_counts = df[col].value_counts().nlargest(10)  # Top 10 categories
        fig = px.bar(x=value_counts.index, y=value_counts.values)
        fig.update_layout(title=f"Top Categories in {col}",
                         xaxis_title=col,
                         yaxis_title="Count")
        
        # For subsequent visualizations, don't include Plotly.js again
        include_js = 'cdn' if not plotlyjs_included else False
        visualizations.append({
            'title': f"Bar Chart: Top Categories in {col}",
            'plot': fig.to_html(full_html=False, include_plotlyjs=include_js),
            'description': f"Bar chart showing the distribution of top categories in {col}."
        })
        plotlyjs_included = True
    
    print(f"Total visualizations created: {len(visualizations)}")
    return visualizations

def generate_insights(df1, df2=None):
    """Generate insights about the data"""
    insights = []
    
    # Basic insights for df1
    insights.append(f"Dataset 1 contains {df1.shape[0]} rows and {df1.shape[1]} columns.")
    
    # Missing values
    missing = df1.isnull().sum().sum()
    if missing > 0:
        missing_pct = (missing / (df1.shape[0] * df1.shape[1])) * 100
        insights.append(f"Dataset 1 has {missing} missing values ({missing_pct:.2f}% of all data).")
    else:
        insights.append("Dataset 1 has no missing values.")
    
    # Numeric columns
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights.append(f"Dataset 1 has {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}.")
        
        # Check for correlation
        if len(numeric_cols) >= 2:
            corr = df1[numeric_cols].corr()
            # Get the top correlation pair
            corr_values = corr.unstack()
            corr_values = corr_values[corr_values < 1.0]  # Remove self-correlations
            if not corr_values.empty:
                max_corr = corr_values.abs().max()
                if max_corr > 0.7:
                    max_pair = corr_values.abs().idxmax()
                    insights.append(f"Strong correlation detected between {max_pair[0]} and {max_pair[1]} (r={corr_values.loc[max_pair]:.2f}).")
    
    # Categorical columns
    cat_cols = df1.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        insights.append(f"Dataset 1 has {len(cat_cols)} categorical columns: {', '.join(cat_cols)}.")
        
        # Check for cardinality
        for col in cat_cols:
            unique_values = df1[col].nunique()
            if unique_values == 1:
                insights.append(f"Column '{col}' has only one unique value and might not be useful for analysis.")
            elif unique_values > 10 and unique_values / len(df1) > 0.9:
                insights.append(f"Column '{col}' has high cardinality ({unique_values} values) and might be an ID column.")
    
    # Compare datasets if df2 is provided
    if df2 is not None:
        common_cols = list(set(df1.columns) & set(df2.columns))
        insights.append(f"The two datasets share {len(common_cols)} common columns.")
        
        # Row count comparison
        row_diff = abs(len(df1) - len(df2))
        row_diff_pct = (row_diff / max(len(df1), len(df2))) * 100
        if row_diff > 0:
            insights.append(f"The datasets differ by {row_diff} rows ({row_diff_pct:.2f}% difference).")
        
        # Compare common numeric columns
        common_numeric = [col for col in common_cols if col in df1.select_dtypes(include=[np.number]).columns
                         and col in df2.select_dtypes(include=[np.number]).columns]
        
        if common_numeric:
            for col in common_numeric:
                mean_diff = abs(df1[col].mean() - df2[col].mean())
                max_mean = max(abs(df1[col].mean()), abs(df2[col].mean()))
                if max_mean > 0:
                    mean_diff_pct = (mean_diff / max_mean) * 100
                    if mean_diff_pct > 10:  # If means differ by more than 10%
                        insights.append(f"The mean values for '{col}' differ by {mean_diff_pct:.2f}% between datasets.")
    
    return insights

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file1' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file1 = request.files['file1']
    if file1.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if not allowed_file(file1.filename):
        flash('Invalid file type. Please upload CSV or Excel files.')
        return redirect(request.url)
    
    try:
        print(f"Processing file: {file1.filename}")
        # Read the first file
        df1 = read_file(file1)
        df1_html = df1.head(10).to_html(classes="table table-striped", index=False)
        summary1_html = generate_summary_stats(df1)
        
        # Check if a second file was uploaded
        df2 = None
        df2_html = None
        summary2_html = None
        comparison_results = None
        
        if 'file2' in request.files and request.files['file2'].filename != '':
            file2 = request.files['file2']
            print(f"Processing second file: {file2.filename}")
            if allowed_file(file2.filename):
                df2 = read_file(file2)
                df2_html = df2.head(10).to_html(classes="table table-striped", index=False)
                summary2_html = generate_summary_stats(df2)
                comparison_results = compare_datasets(df1, df2)
        
        # Generate visualizations
        visualizations = generate_visualizations(df1)
        
        # Generate correlation plot if requested
        correlation_plot = None
        if 'correlation' in request.form:
            correlation_plot = generate_correlation_plot(df1)
        
        # Detect outliers if requested
        outliers = None
        if 'outlier_detection' in request.form:
            outliers = detect_outliers(df1)
        
        # Generate insights
        insights = generate_insights(df1, df2)
        
        return render_template('visualization.html', 
                              analysis_results=True,
                              df1_html=df1_html,
                              df2_html=df2_html,
                              summary1_html=summary1_html,
                              summary2_html=summary2_html,
                              visualizations=visualizations,
                              correlation_plot=correlation_plot,
                              outliers=outliers,
                              comparison_results=comparison_results,
                              insights=insights)
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error processing file: {str(e)}")
        print(error_details)
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('visualization'))

if __name__ == '__main__':
    app.run(debug=True)