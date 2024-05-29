import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

## TODOS:
# Add in more documentation
# Add in where statement option for data set
# Add in "by" variable option
# Add in optionality for exporting to CSV
# Expand example

# Define the number of rows
num_rows = 100

# Create a dictionary with column names and dummy data
data = {
    'id': np.arange(1, num_rows + 1),
    'name': [f'Name{i}' for i in range(1, num_rows + 1)],
    'age': np.random.randint(18, 70, size=num_rows),
    'salary': np.random.randint(30000, 100000, size=num_rows),
    'join_date': pd.date_range(start='2020-01-01', periods=num_rows, freq='D')
}

# Create a DataFrame
df = pd.DataFrame(data)
# Age categories we will use
bins = [0, 18, 35, 50, 65, np.inf]
labels = ['<18', '18-34', '35-49', '50-64', '65+']
df['age_category'] = pd.cut(df['age'], bins = bins, labels=labels, right=False)

# Assign each person to a specific city
cities = ['New York', 'Los Angeles', 'Chicago']
df['city'] = np.random.choice(cities, size=len(df))

# Display the first few rows of the DataFrame
print(df.head())
df.describe


# Function to run regressions based on outcome types
def run_regressions(data, bin_outcomes, count_outcomes, cost_outcomes, cohort, covariates, class_vars, cohort_ref):
    results = []

    all_outcomes = bin_outcomes + count_outcomes + cost_outcomes
    for outcome in all_outcomes:
        formula = f"{outcome} ~ {' + '.join(cohort)} + {' + '.join(covariates)}"
        
        if outcome in bin_outcomes:
            model = smf.logit(formula, data=data).fit()
        elif outcome in count_outcomes:
            model = smf.poisson(formula, data=data).fit()
        elif outcome in cost_outcomes:
            model = smf.glm(formula, data=data, family=sm.families.Gamma(link=sm.families.links.log())).fit()
        
        summary = model.summary2().tables[1].reset_index()
        summary['outcome'] = outcome
        results.append(summary)
    
    final_output = pd.concat(results, axis=0)
    return final_output

# Function to iterate through specifications and run regressions
def covar_lists(spec_data, data_out):
    all_results = []
    for i, spec in spec_data.iterrows():

        result = run_regressions(spec['data'], spec['bin_outcomes'], spec['count_outcomes'], 
        spec['cost_outcomes'], spec['cohort'], spec['covariates'], 
        spec['class_vars'], spec['cohort_ref'])
        result['spec'] = spec['desc']
        all_results.append(result)
    
    final_results = pd.concat(all_results, axis=0)
    #TODO make to_csv optional
    final_results.to_csv(data_out, index=False)
    return final_results

# Example usage
# Define your specifications DataFrame
specifications = pd.DataFrame({
    'data': [df],
    'bin_outcomes': [[]],
    'count_outcomes': [[]],
    'cost_outcomes': [['salary']],
    'cohort': [['age_category']],
    'covariates': [['age', 'age_category']],
    'class_vars': [['age_category']],
    'cohort_ref': [['city(ref="New York")']],
    'desc': ['test']
})

# Your data
# data = pd.read_csv("your_data.csv")

# Run the covar_lists function
outPath = "C:\\Users\\luke\\Documents\\Temporary\\"
final_results = covar_lists(specifications, outPath + "data_out.csv")
#test = run_regressions(df, bin_outcomes, count_outcomes, cost_outcomes, cohort, covariates, class_vars, cohort_ref)
print(final_results)


for i, spec in specifications.iterrows():
    result = run_regressions(spec['data'], spec['bin_outcomes'], spec['count_outcomes'], 
                             spec['cost_outcomes'], spec['cohort'], spec['covariates'], 
                             spec['class_vars'], spec['cohort_ref'])
    result['spec'] = spec['desc']
    print(result)