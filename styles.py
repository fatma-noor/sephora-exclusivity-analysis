import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display

# =============================================================================
# COLOUR PALETTE
# =============================================================================

COLORS = ["#80FFDB", "#72EFDD", "#64DFDF", "#56CFE1", "#48BFE3", "#4EA8DE", "#5390D9", "#5E60CE"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom', COLORS)
EXCL_COLORS = ['#f4a582', '#a8d5e5']  # Non-exclusive, Exclusive


# =============================================================================
# AXIS AND PLOT STYLING
# =============================================================================

def style_axis(ax, title='', xlabel='', ylabel=''):
    """Apply consistent styling to matplotlib axis"""
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1a1a2e', pad=15)
    ax.set_xlabel(xlabel, fontsize=11, color='#444', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, color='#444', labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ddd')
    ax.spines['bottom'].set_color('#ddd')
    ax.tick_params(labelsize=10)


def styled_boxplot(ax, df_box, y, x='sephora_exclusive', title='', xlabel='Exclusive',
                   ylabel='', order=(False, True), width=0.5, showfliers=True,
                   line_w=1.6, median_w=2.2, flier_size=3.2, flier_alpha=0.35):
    """Create styled boxplot comparing exclusive vs non-exclusive"""
    pal = {order[0]: COLORS[3], order[1]: COLORS[0]}

    box_kws = dict(linewidth=line_w, edgecolor='#111111')
    whisker_kws = dict(linewidth=line_w, color='#111111')
    cap_kws = dict(linewidth=line_w, color='#111111')
    median_kws = dict(linewidth=median_w, color='#111111')
    flier_kws = dict(marker='o', markersize=flier_size,
                     markerfacecolor='#111111', markeredgecolor='#111111',
                     alpha=flier_alpha)

    ax.set_facecolor('#ffffff')

    sns.boxplot(
        data=df_box, x=x, y=y,
        hue=x, order=list(order), hue_order=list(order),
        palette=pal, width=width, ax=ax, legend=False,
        showfliers=showfliers,
        boxprops=box_kws, whiskerprops=whisker_kws, capprops=cap_kws,
        medianprops=median_kws, flierprops=flier_kws
    )

    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not exclusive', 'Exclusive'])


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_exclusive(df_in, col='sephora_exclusive'):
    """Clean and convert exclusivity column to boolean"""
    if col not in df_in.columns:
        raise KeyError(f"Column '{col}' not found in df")

    s = (df_in[col]
         .astype(str).str.strip().str.lower()
         .replace({'nan': np.nan, 'none': np.nan, '': np.nan})
         .map({'true': True, 'false': False, '1': True, '0': False,
               'yes': True, 'no': False, 'y': True, 'n': False}))

    df_box = df_in.copy()
    df_box[col] = s
    df_box = df_box[df_box[col].notna()].copy()
    df_box[col] = df_box[col].astype(bool)
    return df_box


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_pval(p):
    """Format p-value for display"""
    if p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return 'p < 0.01'
    elif p < 0.05:
        return 'p < 0.05'
    else:
        return f'p = {p:.3f}'


# =============================================================================
# STYLED TABLES
# =============================================================================

def styled_table(df_in, caption=None, precision=3, fmt=None, mode='full',
                 max_colwidth=40, max_rows=None):
    """
    Create styled pandas table.

    Parameters
    ----------
    df_in : DataFrame
        Input dataframe to style
    caption : str, optional
        Table caption
    precision : int
        Decimal precision for numeric values
    fmt : dict, optional
        Custom format dictionary
    mode : str
        'preview' for compact display, 'full' for gradient styling
    max_colwidth : int
        Maximum column width before truncation
    max_rows : int, optional
        Maximum rows to display
    """
    df_show = df_in.copy()
    if max_rows is not None:
        df_show = df_show.head(max_rows)

    def _truncate(x):
        if isinstance(x, str) and len(x) > max_colwidth:
            return x[:max_colwidth-1] + '...'
        return x

    if mode == 'preview':
        df_show = df_show.applymap(_truncate)

    sty = df_show.style

    if fmt is not None:
        sty = sty.format(fmt)
    else:
        sty = sty.format(precision=precision, thousands=',')

    if mode == 'full':
        sty = sty.background_gradient(cmap=CUSTOM_CMAP, axis=None)

    preview_css = []
    if mode == 'preview':
        preview_css = [
            {'selector': 'td', 'props': [
                ('max-width', '220px'), ('white-space', 'nowrap'),
                ('overflow', 'hidden'), ('text-overflow', 'ellipsis'),
                ('vertical-align', 'top')
            ]},
            {'selector': 'th', 'props': [('white-space', 'nowrap')]},
            {'selector': '', 'props': [
                ('display', 'block'), ('overflow-x', 'auto'), ('white-space', 'nowrap')
            ]},
        ]

    sty = (sty
        .set_properties(**{
            'font-family': 'Segoe UI, sans-serif',
            'font-size': '12px',
            'text-align': 'center',
            'padding': '4px 8px',
            'line-height': '1.15'
        })
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background', '#5E60CE'),
                ('color', 'white'),
                ('font-weight', '600'), ('font-size', '12px'),
                ('text-align', 'left'), ('padding', '12px 15px'),
                ('border', 'none'), ('line-height', '1.15'),
                ('position', 'sticky'), ('top', '0')
            ]},
            {'selector': 'th.row_heading', 'props': [
                ('background', '#5E60CE'),
                ('color', '#FFFFFF'),
                ('font-weight', '600'), ('text-align', 'left'),
                ('padding', '12px 15px'), ('line-height', '1.15')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'), ('border-radius', '10px'),
                ('overflow', 'hidden'), ('box-shadow', '0 4px 15px rgba(94, 96, 206, 0.1)'),
                ('background-color', '#ffffff')
            ]},
            {'selector': 'td:hover', 'props': [
                ('background-color', '#f0f0ff !important'), ('cursor', 'pointer')
            ]},
            {'selector': 'caption', 'props': [
                ('caption-side', 'top'), ('font-size', '18px'),
                ('font-weight', 'bold'), ('color', '#1a1a2e'),
                ('padding', '12px 0'), ('text-align', 'left')
            ]},
            *preview_css
        ])
    )

    if caption:
        sty = sty.set_caption(caption)

    return sty


# =============================================================================
# HTML DISPLAY COMPONENTS
# =============================================================================

def styled_shape(df):
    """Display dataframe shape in a styled gradient box"""
    html = f'''
    <div style="font-family: 'Segoe UI', sans-serif; display: inline-flex; gap: 20px;
                padding: 20px 30px; background: linear-gradient(135deg, #5E60CE 0%, #48BFE3 100%);
                border-radius: 12px; box-shadow: 0 4px 15px rgba(94, 96, 206, 0.3);">
        <div style="text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: white;">{df.shape[0]:,}</div>
            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8); text-transform: uppercase; letter-spacing: 1px;">Rows</div>
        </div>
        <div style="width: 1px; background: rgba(255,255,255,0.3);"></div>
        <div style="text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: white;">{df.shape[1]}</div>
            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8); text-transform: uppercase; letter-spacing: 1px;">Columns</div>
        </div>
    </div>
    '''
    display(HTML(html))


def styled_missing(df):
    """Display missing values summary table"""
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str).values,
        'Missing': df.isnull().sum().values,
        'Percent': (df.isnull().sum().values / len(df) * 100).round(1)
    }).sort_values('Missing', ascending=False)

    rows_html = ""
    for _, row in missing_df.iterrows():
        pct = row['Percent']
        color = '#5E60CE' if pct > 30 else ('#48BFE3' if pct > 10 else '#80FFDB')
        rows_html += f'''
        <tr>
            <td style="color: #5E60CE; font-family: Consolas, monospace; font-weight: 600;">{row['Column']}</td>
            <td style="text-align: center;">{row['Data Type']}</td>
            <td style="text-align: center;">{int(row['Missing']):,}</td>
            <td style="color: {color}; font-weight: 500; text-align: center;">{pct}%</td>
        </tr>
        '''

    html = f'''
    <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px;
                  overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06); font-family: 'Segoe UI', sans-serif;">
        <thead>
            <tr style="background: linear-gradient(135deg, #5E60CE 0%, #48BFE3 100%);">
                <th style="color: white; padding: 12px 15px; text-align: left; font-weight: 600;">Column</th>
                <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Data Type</th>
                <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Missing</th>
                <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Percent</th>
            </tr>
        </thead>
        <tbody style="font-size: 0.9rem;">
            {rows_html}
        </tbody>
    </table>
    '''
    display(HTML(html))


def styled_ttest_table(results_list, one_sided=False):
    """
    Display t-test results in a styled table.

    Parameters
    ----------
    results_list : list of dict
        Each dict should have keys: 'variable', 't_stat', 'p_value', 'significant'
    one_sided : bool
        Whether the test is one-sided
    """
    rows_html = ""
    for r in results_list:
        sig_color = "#80FFDB" if r['significant'] else EXCL_COLORS[1]
        sig_text = "Yes" if r['significant'] else "No"
        p_fmt = format_pval(r['p_value']) if isinstance(r['p_value'], float) else r['p_value']

        h0 = "mu_excl = mu_non"
        h1 = "mu_excl > mu_non" if one_sided else "mu_excl != mu_non"

        rows_html += f'''
        <tr>
            <td style="font-weight: 600; color: #5E60CE; text-align: left; padding: 10px 15px;">{r['variable']}</td>
            <td style="text-align: left; padding: 10px 15px;">{h0}</td>
            <td style="text-align: left; padding: 10px 15px;">{h1}</td>
            <td style="text-align: left; padding: 10px 15px; font-weight: 500;">{r['t_stat']:.3f}</td>
            <td style="text-align: left; padding: 10px 15px; font-weight: 500;">{p_fmt}</td>
            <td style="text-align: center; padding: 10px 15px; font-weight: 700;
                       color: {'#2e7d32' if r['significant'] else '#c62828'};
                       background: {sig_color}; border-radius: 4px;">{sig_text}</td>
        </tr>
        '''

    test_type = "one-sided" if one_sided else "two-sided"
    html = f'''
    <div style="font-family: 'Segoe UI', sans-serif; margin-top: 20px;">
        <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 12px;
                    padding-bottom: 8px; border-bottom: 2px solid #5E60CE;">T-Test Results (alpha = 0.05, {test_type})</div>
        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px;
                      overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06);">
            <thead>
                <tr style="background: linear-gradient(135deg, #5E60CE 0%, #48BFE3 100%);">
                    <th style="color: white; padding: 12px 15px; text-align: left; font-weight: 600;">Variable</th>
                    <th style="color: white; padding: 12px 15px; text-align: left; font-weight: 600;">H0 (Null)</th>
                    <th style="color: white; padding: 12px 15px; text-align: left; font-weight: 600;">H1 (Alternative)</th>
                    <th style="color: white; padding: 12px 15px; text-align: left; font-weight: 600;">t-statistic</th>
                    <th style="color: white; padding: 12px 15px; text-align: left; font-weight: 600;">p-value</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Significant</th>
                </tr>
            </thead>
            <tbody style="font-size: 0.9rem;">
                {rows_html}
            </tbody>
        </table>
    </div>
    '''
    display(HTML(html))


def styled_engagement_stats(stats_data, excl_df, non_excl_df):
    """
    Display engagement stats boxes in a row matching the plots.

    Parameters
    ----------
    stats_data : list of tuples
        Each tuple is (column_name, display_label)
    excl_df : DataFrame
        Exclusive products dataframe
    non_excl_df : DataFrame
        Non-exclusive products dataframe
    """
    html = '''
    <div style="font-family: 'Segoe UI', sans-serif; display: flex; gap: 20px; justify-content: space-around; margin-top: 10px;">
    '''

    for col, label in stats_data:
        excl_mean = excl_df[col].mean()
        excl_med = excl_df[col].median()
        non_excl_mean = non_excl_df[col].mean()
        non_excl_med = non_excl_df[col].median()

        html += f'''
        <div style="display: flex; gap: 10px;">
            <div style="background: {EXCL_COLORS[1]}; padding: 12px 20px; border-radius: 10px; text-align: center; min-width: 140px;">
                <div style="font-size: 0.75rem; font-weight: 600; color: white; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Exclusive</div>
                <div style="display: flex; gap: 15px; justify-content: center;">
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: white;">{excl_mean:.2f}</div>
                        <div style="font-size: 0.65rem; color: rgba(255,255,255,0.8);">Mean</div>
                    </div>
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: white;">{excl_med:.2f}</div>
                        <div style="font-size: 0.65rem; color: rgba(255,255,255,0.8);">Median</div>
                    </div>
                </div>
            </div>
            <div style="background: {EXCL_COLORS[0]}; padding: 12px 20px; border-radius: 10px; text-align: center; min-width: 140px;">
                <div style="font-size: 0.75rem; font-weight: 600; color: white; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Non-Exclusive</div>
                <div style="display: flex; gap: 15px; justify-content: center;">
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: white;">{non_excl_mean:.2f}</div>
                        <div style="font-size: 0.65rem; color: rgba(255,255,255,0.8);">Mean</div>
                    </div>
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: white;">{non_excl_med:.2f}</div>
                        <div style="font-size: 0.65rem; color: rgba(255,255,255,0.8);">Median</div>
                    </div>
                </div>
            </div>
        </div>
        '''

    html += "</div>"
    display(HTML(html))


def styled_price_stats(excl_mean, excl_med, non_excl_mean, non_excl_med):
    """Display price comparison stats boxes"""
    html = f'''
    <div style="font-family: 'Segoe UI', sans-serif; display: flex; gap: 20px; justify-content: center; margin-top: 15px;">
        <div style="background: {EXCL_COLORS[1]}; padding: 15px 25px; border-radius: 10px; text-align: center; min-width: 180px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: white; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Exclusive</div>
            <div style="display: flex; gap: 20px; justify-content: center;">
                <div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: white;">{excl_mean:.2f}</div>
                    <div style="font-size: 0.7rem; color: rgba(255,255,255,0.8);">Mean</div>
                </div>
                <div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: white;">{excl_med:.2f}</div>
                    <div style="font-size: 0.7rem; color: rgba(255,255,255,0.8);">Median</div>
                </div>
            </div>
        </div>
        <div style="background: {EXCL_COLORS[0]}; padding: 15px 25px; border-radius: 10px; text-align: center; min-width: 180px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: white; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Non-Exclusive</div>
            <div style="display: flex; gap: 20px; justify-content: center;">
                <div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: white;">{non_excl_mean:.2f}</div>
                    <div style="font-size: 0.7rem; color: rgba(255,255,255,0.8);">Mean</div>
                </div>
                <div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: white;">{non_excl_med:.2f}</div>
                    <div style="font-size: 0.7rem; color: rgba(255,255,255,0.8);">Median</div>
                </div>
            </div>
        </div>
    </div>
    '''
    display(HTML(html))


# =============================================================================
# CLUSTERING DISPLAY FUNCTIONS
# =============================================================================

def styled_cluster_summary(cluster_stats_df):
    """Display cluster summary statistics in styled table format"""
    rows_html = ""
    cluster_colors_hex = ["#5E60CE", "#80FFDB", "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"]

    for idx, row in cluster_stats_df.iterrows():
        color = cluster_colors_hex[int(idx) % len(cluster_colors_hex)]
        rows_html += f'''
        <tr>
            <td style="text-align: center;">
                <span style="display: inline-block; width: 30px; height: 30px; background: {color};
                             border-radius: 50%; color: white; font-weight: 700; line-height: 30px;">{int(idx)}</span>
            </td>
            <td style="text-align: center; font-weight: 600;">{int(row['n_products']):,}</td>
            <td style="text-align: center;">{row['pct_of_total']:.1f}%</td>
            <td style="text-align: center;">${row['avg_price']:.0f}</td>
            <td style="text-align: center;">{row['avg_loves']:,.0f}</td>
            <td style="text-align: center;">{row['avg_rating']:.2f}</td>
            <td style="text-align: center;">{row['avg_reviews']:.0f}</td>
            <td style="text-align: center; font-weight: 600; color: {'#2e7d32' if row['pct_exclusive'] > 30 else '#1a1a2e'};">{row['pct_exclusive']:.1f}%</td>
        </tr>
        '''

    html = f'''
    <div style="font-family: 'Segoe UI', sans-serif; margin-top: 20px;">
        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px;
                      overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06);">
            <thead>
                <tr style="background: linear-gradient(135deg, #5E60CE 0%, #48BFE3 100%);">
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Cluster</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Products</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">% of Total</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Avg Price</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Avg Loves</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Avg Rating</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Avg Reviews</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">% Exclusive</th>
                </tr>
            </thead>
            <tbody style="font-size: 0.9rem;">
                {rows_html}
            </tbody>
        </table>
    </div>
    '''
    display(HTML(html))


def styled_cluster_exclusivity(cluster_excl_df):
    """Display exclusivity breakdown by cluster as styled cards"""
    cluster_colors_hex = ["#5E60CE", "#80FFDB", "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"]

    cards_html = ""
    for _, row in cluster_excl_df.iterrows():
        cluster_id = int(row['cluster'])
        color = cluster_colors_hex[cluster_id % len(cluster_colors_hex)]
        excl_pct = row['pct_exclusive']
        bar_color = EXCL_COLORS[1]

        cards_html += f'''
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
                    border-top: 4px solid {color}; min-width: 140px; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 700; color: {color};">Cluster {cluster_id}</div>
            <div style="margin: 10px 0;">
                <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="background: {bar_color}; height: 100%; width: {excl_pct}%;"></div>
                </div>
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e;">{excl_pct:.1f}% Exclusive</div>
            <div style="font-size: 0.8rem; color: #666;">{int(row['n_exclusive']):,} of {int(row['n_products']):,}</div>
        </div>
        '''

    html = f'''
    <div style="font-family: 'Segoe UI', sans-serif; display: flex; gap: 15px; flex-wrap: wrap; justify-content: center; margin-top: 15px;">
        {cards_html}
    </div>
    '''
    display(HTML(html))


# =============================================================================
# NOTIFICATION / ALERT BOXES
# =============================================================================

def styled_note(message, note_type='info'):
    """
    Display a styled notification box.

    Parameters
    ----------
    message : str
        The message to display
    note_type : str
        'info', 'warning', or 'success'
    """
    colors_map = {
        'info': ('#5E60CE', '#f8f9ff'),
        'warning': ('#f4a582', '#fff8f5'),
        'success': ('#80FFDB', '#f5fffc')
    }
    border_color, bg_color = colors_map.get(note_type, colors_map['info'])

    html = f'''
    <div style="font-family: 'Segoe UI', sans-serif; padding: 20px; background: {bg_color};
                border-left: 4px solid {border_color}; border-radius: 8px; color: #1a1a2e; margin: 10px 0;">
        {message}
    </div>
    '''
    display(HTML(html))


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("styles.py loaded successfully!")
    print(f"COLORS: {COLORS}")
    print(f"EXCL_COLORS: {EXCL_COLORS}")
    print("All functions available.")


# =============================================================================
# REGRESSION DISPLAY FUNCTIONS
# =============================================================================

def styled_ols_result(pct_effect, excl_pval, pct_ci_lower, pct_ci_upper):
    """Display OLS regression key finding as styled cards."""
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; margin: 20px 0;">
        <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 15px; 
                    padding-bottom: 8px; border-bottom: 2px solid #5E60CE;">
            📊 OLS Regression: Key Finding
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="background: linear-gradient(135deg, #5E60CE 0%, #48BFE3 100%); 
                        padding: 20px; border-radius: 10px; min-width: 200px; text-align: center;">
                <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem; text-transform: uppercase; 
                            letter-spacing: 1px;">Exclusivity Effect</div>
                <div style="color: white; font-size: 2rem; font-weight: 700; margin: 8px 0;">
                    {'+' if pct_effect > 0 else ''}{pct_effect:.1f}%
                </div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">
                    more loves for exclusive products
                </div>
            </div>
            
            <div style="background: {'#80FFDB' if excl_pval < 0.05 else '#f4a582'}; 
                        padding: 20px; border-radius: 10px; min-width: 200px; text-align: center;">
                <div style="color: #1a1a2e; font-size: 0.8rem; text-transform: uppercase; 
                            letter-spacing: 1px;">Statistical Significance</div>
                <div style="color: #1a1a2e; font-size: 2rem; font-weight: 700; margin: 8px 0;">
                    p {'< 0.001' if excl_pval < 0.001 else f'= {excl_pval:.4f}'}
                </div>
                <div style="color: #1a1a2e; font-size: 0.85rem;">
                    {'Highly Significant ✓' if excl_pval < 0.001 else 'Significant ✓' if excl_pval < 0.05 else 'Not Significant ✗'}
                </div>
            </div>
            
            <div style="background: #f0f0ff; padding: 20px; border-radius: 10px; min-width: 200px; text-align: center;">
                <div style="color: #5E60CE; font-size: 0.8rem; text-transform: uppercase; 
                            letter-spacing: 1px;">95% Confidence Interval</div>
                <div style="color: #1a1a2e; font-size: 1.5rem; font-weight: 700; margin: 8px 0;">
                    [{pct_ci_lower:.1f}%, {pct_ci_upper:.1f}%]
                </div>
                <div style="color: #636e72; font-size: 0.85rem;">
                    Effect range
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; 
                    border-left: 4px solid #5E60CE;">
            <strong>Interpretation:</strong> After controlling for price, rating, review volume, 
            product flags, and category fixed effects, exclusive products receive approximately 
            <strong>{pct_effect:.1f}%</strong> more loves than comparable non-exclusive products. 
            This effect is statistically {'highly ' if excl_pval < 0.001 else ''}significant 
            (p {'< 0.001' if excl_pval < 0.001 else f'= {excl_pval:.4f}'}).
        </div>
    </div>
    """
    display(HTML(html))


def styled_sentiment_summary(excl_pos, excl_neg, non_pos, non_neg):
    """Display sentiment word summary as styled output."""
    excl_pct = 100 * excl_pos / (excl_pos + excl_neg) if (excl_pos + excl_neg) > 0 else 0
    non_pct = 100 * non_pos / (non_pos + non_neg) if (non_pos + non_neg) > 0 else 0
    excl_ratio = excl_pos / max(excl_neg, 1)
    non_ratio = non_pos / max(non_neg, 1)
    
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; margin: 20px 0;">
        <div style="font-size: 1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 15px; 
                    padding-bottom: 8px; border-bottom: 2px solid #5E60CE;">
            Sentiment Word Summary
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="background: {EXCL_COLORS[1]}; padding: 20px; border-radius: 10px; 
                        min-width: 180px; text-align: center;">
                <div style="color: white; font-size: 0.8rem; text-transform: uppercase; 
                            letter-spacing: 1px; margin-bottom: 10px;">Exclusive</div>
                <div style="color: white; font-size: 1.8rem; font-weight: 700;">{excl_pct:.1f}%</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">positive words</div>
                <div style="color: white; font-size: 0.9rem; margin-top: 8px;">
                    Ratio: {excl_ratio:.1f} pos/neg
                </div>
            </div>
            
            <div style="background: {EXCL_COLORS[0]}; padding: 20px; border-radius: 10px; 
                        min-width: 180px; text-align: center;">
                <div style="color: white; font-size: 0.8rem; text-transform: uppercase; 
                            letter-spacing: 1px; margin-bottom: 10px;">Non-Exclusive</div>
                <div style="color: white; font-size: 1.8rem; font-weight: 700;">{non_pct:.1f}%</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">positive words</div>
                <div style="color: white; font-size: 0.9rem; margin-top: 8px;">
                    Ratio: {non_ratio:.1f} pos/neg
                </div>
            </div>
            
            <div style="background: #f0f0ff; padding: 20px; border-radius: 10px; 
                        min-width: 180px; text-align: center;">
                <div style="color: #5E60CE; font-size: 0.8rem; text-transform: uppercase; 
                            letter-spacing: 1px; margin-bottom: 10px;">Difference</div>
                <div style="color: #1a1a2e; font-size: 1.8rem; font-weight: 700;">
                    {excl_pct - non_pct:+.1f}%
                </div>
                <div style="color: #636e72; font-size: 0.8rem;">Δ positive %</div>
                <div style="color: #636e72; font-size: 0.9rem; margin-top: 8px;">
                    {'Negligible' if abs(excl_pct - non_pct) < 2 else 'Small' if abs(excl_pct - non_pct) < 5 else 'Notable'}
                </div>
            </div>
        </div>
    </div>
    """
    display(HTML(html))


# =============================================================================
# SENTIMENT ANALYSIS HELPERS
# =============================================================================

# Emotion whitelist for vocabulary analysis
EMOTION_WHITELIST = {
    # Positive emotions
    'love', 'loved', 'loving', 'amazing', 'awesome', 'excellent', 'fantastic',
    'wonderful', 'perfect', 'great', 'good', 'best', 'beautiful', 'favorite',
    'obsessed', 'addicted', 'hooked', 'incredible', 'brilliant', 'superb',
    'holy', 'grail', 'miracle', 'stunning', 'gorgeous', 'lovely', 'nice',
    # Negative emotions
    'hate', 'hated', 'terrible', 'horrible', 'awful', 'worst', 'bad', 'poor',
    'disappointed', 'disappointing', 'underwhelming', 'mediocre', 'meh',
    'waste', 'overpriced', 'overhyped', 'regret', 'unfortunately', 'sadly',
    # Product quality - positive
    'smooth', 'soft', 'hydrating', 'moisturizing', 'nourishing', 'soothing',
    'glowing', 'radiant', 'flawless', 'lightweight', 'gentle', 'effective',
    'worth', 'recommend', 'repurchase', 'rebuy', 'affordable',
    'pigmented', 'blendable', 'buildable', 'creamy', 'silky', 'velvety',
    'luxurious', 'refreshing', 'calming', 'healing',
    # Product quality - negative
    'broke', 'breakout', 'breakouts', 'irritated', 'irritating', 'burning',
    'sticky', 'greasy', 'oily', 'heavy', 'cakey', 'patchy', 'streaky', 
    'flaky', 'drying', 'harsh', 'stinging', 'itchy', 'redness',
    'return', 'returned', 'refund',
    # Descriptive neutral
    'lasting', 'coverage', 'natural', 'dewy', 'matte', 'sheer', 'full',
    'subtle', 'strong', 'light', 'dark', 'shiny', 'glowy'
}

POSITIVE_WORDS = {'love', 'loved', 'loving', 'amazing', 'awesome', 'excellent', 'fantastic',
                  'wonderful', 'perfect', 'great', 'good', 'best', 'beautiful', 'favorite',
                  'obsessed', 'incredible', 'brilliant', 'stunning', 'gorgeous', 'lovely',
                  'nice', 'recommend', 'repurchase', 'effective', 'worth', 'gentle', 'smooth',
                  'soft', 'hydrating', 'moisturizing', 'glowing', 'radiant', 'flawless',
                  'lightweight', 'luxurious', 'refreshing', 'soothing', 'calming', 'nourishing'}

NEGATIVE_WORDS = {'hate', 'hated', 'terrible', 'horrible', 'awful', 'worst', 'bad', 'poor',
                  'disappointed', 'disappointing', 'underwhelming', 'mediocre', 'waste',
                  'overpriced', 'overhyped', 'regret', 'broke', 'breakout', 'breakouts',
                  'irritated', 'irritating', 'burning', 'sticky', 'greasy', 'oily', 'heavy',
                  'cakey', 'patchy', 'streaky', 'flaky', 'drying', 'harsh', 'return', 'returned'}


def extract_emotions(text):
    """Extract emotion words from text using whitelist."""
    import re
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return [w for w in words if w in EMOTION_WHITELIST]


def get_sentiment(text):
    """Get TextBlob sentiment polarity for text."""
    from textblob import TextBlob
    if pd.isna(text) or len(str(text).strip()) == 0:
        return np.nan
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return np.nan


# =============================================================================
# PLOT HELPER
# =============================================================================

def create_figure(nrows=1, ncols=1, figsize=None, dpi=150):
    """Create figure with consistent white background styling."""
    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#ffffff', dpi=dpi)
    
    # Set white background for all axes
    if nrows * ncols == 1:
        axes.set_facecolor('#ffffff')
    else:
        for ax in np.array(axes).flatten():
            ax.set_facecolor('#ffffff')
    
    return fig, axes

# =============================================================================
# BRAND ANALYSIS FUNCTIONS
# =============================================================================

def styled_brand_stats(top_brands_df):
    """Display top brands summary statistics."""
    rows_html = ""
    for i, (idx, row) in enumerate(top_brands_df.iterrows()):
        excl_pct = row['exclusive_count'] / row['total_count'] * 100
        rows_html += f'''
        <tr>
            <td style="font-weight: 600; color: #5E60CE; text-align: left; padding: 10px 15px;">{idx}</td>
            <td style="text-align: center; padding: 10px 15px;">{int(row['total_count']):,}</td>
            <td style="text-align: center; padding: 10px 15px;">{int(row['exclusive_count']):,}</td>
            <td style="text-align: center; padding: 10px 15px;">{int(row['non_exclusive_count']):,}</td>
            <td style="text-align: center; padding: 10px 15px; font-weight: 600; color: {'#5E60CE' if excl_pct > 30 else '#1a1a2e'};">{excl_pct:.1f}%</td>
        </tr>
        '''
    
    html = f'''
    <div style="font-family: 'Segoe UI', sans-serif; margin-top: 20px;">
        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px;
                      overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06);">
            <thead>
                <tr style="background: linear-gradient(135deg, #5E60CE 0%, #48BFE3 100%);">
                    <th style="color: white; padding: 12px 15px; text-align: left; font-weight: 600;">Brand</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Total Products</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Exclusive</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">Non-Exclusive</th>
                    <th style="color: white; padding: 12px 15px; text-align: center; font-weight: 600;">% Exclusive</th>
                </tr>
            </thead>
            <tbody style="font-size: 0.9rem;">
                {rows_html}
            </tbody>
        </table>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# UPDATE IMPORTS IN NOTEBOOK
# =============================================================================
"""
After adding these functions to styles.py, update your notebook import:

from styles import (
    COLORS, CUSTOM_CMAP, EXCL_COLORS,
    style_axis, styled_boxplot, prepare_exclusive, format_pval,
    styled_table, styled_shape, styled_missing, styled_ttest_table,
    styled_engagement_stats, styled_price_stats,
    styled_cluster_summary, styled_cluster_exclusivity, styled_note,
    # NEW:
    styled_ols_result, styled_sentiment_summary,
    EMOTION_WHITELIST, POSITIVE_WORDS, NEGATIVE_WORDS,
    extract_emotions, get_sentiment, create_figure
)
"""