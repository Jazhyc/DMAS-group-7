import pandas as pd

agent_distribution_dict = {
    'random': 'Random',
    'bipart': 'Bipartisan',
}

def csv_to_latex_table(df):
    # Process index column
    df[['Model Type', 'Expert Allocation', 'Agent Allocation', '_', 'Open Mindedness']] = df['index'].str.split('_', expand=True)
    df = df.drop(['index', '_'], axis=1)
   
    # In Model Type replace values with human readable names
    df['Model Type'] = df['Model Type'].replace({
        'wm': 'Weighted Median',
        'dg': 'De Groot'
    })
   
    df['Expert Allocation'] = df['Expert Allocation'].replace(agent_distribution_dict)
    df['Agent Allocation'] = df['Agent Allocation'].replace(agent_distribution_dict)
   
    # Create combined satisfaction column
    def combine_satisfaction(row):
        conditions = [row['R1 (Paper) Satisfied'], row['R2 (Paper) Satisfied'],
                      row['R3 (Paper) Satisfied'], row['R4 (Paper) Satisfied']]
        if all(conditions):
            return '\\cellcolor{lightgreen}'
        else:
            failed = [f'R{i+1}' for i, cond in enumerate(conditions) if not cond]
            return f"""\\cellcolor{{lightred}}{{{", ".join(failed)}}}"""
   
    df['Combined Satisfaction'] = df.apply(combine_satisfaction, axis=1)
   
    # Pivot the dataframe to create OPM and NOPM columns
    df_pivot = df.pivot(index=['Model Type', 'Expert Allocation', 'Agent Allocation'], 
                        columns='Open Mindedness', 
                        values='Combined Satisfaction')
    df_pivot.columns.name = None
    df_pivot.reset_index(inplace=True)
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n"
    latex_table += "Model Type & Expert Allocation & Agent Allocation & \\multicolumn{2}{c|}{Open Mindedness Included} \\\\ \\cline{4-5}\n"
    latex_table += " &  &  & True & False \\\\ \\hline\n"
   
    for _, row in df_pivot.iterrows():
        latex_table += f"{row['Model Type']} & {row['Expert Allocation']} & {row['Agent Allocation']} & "
        latex_table += f"{row['opm']} & {row['nopm']} \\\\ \\hline\n"
   
    latex_table += "\\end{tabular}\n\\caption{Experimental Results}\n\\label{tab:results}\n\\end{table}"
   
    return latex_table

# Example usage
csv_data = pd.read_csv('output/summary.csv')
latex_table = csv_to_latex_table(csv_data)
print(latex_table)