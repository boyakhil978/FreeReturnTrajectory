'''Shared grading and scorecard plot for the mission-comparison notebooks.'''

import matplotlib.pyplot as plt

GRADE_COLOURS = {'Close match': '#0ca30c', 'Reasonable': '#fab219', 'Off': '#d03b3b'}

def grade(pct_error):
    if abs(pct_error) < 5:
        return 'Close match'
    if abs(pct_error) < 15:
        return 'Reasonable'
    return 'Off'

def plot_scorecard(ax, names, pct_errors, loose, title, loose_label, x_max=40):
    '''Horizontal bars of |percent error|, coloured and labelled by grade.

    loose : list of bool, hatches bars whose real-world value is only a loose comparison
    x_max : longer bars are clipped and labelled with their true value
    '''
    y = range(len(names))[::-1]
    for yi, pct, is_loose in zip(y, pct_errors, loose):
        ax.barh(yi, min(abs(pct), x_max), height=0.6, color=GRADE_COLOURS[grade(pct)], edgecolor='WHITE',
                hatch='//' if is_loose else None)
        clipped = '  (bar clipped)' if abs(pct) > x_max else ''
        ax.text(min(abs(pct), x_max) + 0.6, yi, f'{pct:+.1f}%  {grade(pct)}{clipped}', va='center', fontsize=9, color='#52514e')

    for limit in (5, 15):
        ax.axvline(limit, color='GREY', ls='--', lw=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlim(0, x_max + 20)
    ax.set_xlabel('|Model − real| / real  (%)')
    ax.set_title(title)

    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=GRADE_COLOURS['Close match'], label='Close match (< 5%)'),
        plt.Rectangle((0, 0), 1, 1, color=GRADE_COLOURS['Reasonable'], label='Reasonable (5–15%)'),
        plt.Rectangle((0, 0), 1, 1, color=GRADE_COLOURS['Off'], label='Off (> 15%)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='WHITE', edgecolor='GREY', hatch='//', label=loose_label),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)
