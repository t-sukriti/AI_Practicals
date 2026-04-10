from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('Cloudy', 'Rain'),
    ('Humidity', 'Rain')
])

# Step 2: Define Conditional Probability Tables (CPT)

# Cloudy
cpd_cloudy = TabularCPD(
    variable='Cloudy',
    variable_card=2,
    values=[[0.5], [0.5]]
)

# Humidity
cpd_humidity = TabularCPD(
    variable='Humidity',
    variable_card=2,
    values=[[0.6], [0.4]]
)

# Rain (depends on Cloudy and Humidity)
cpd_rain = TabularCPD(
    variable='Rain',
    variable_card=2,
    values=[
        [0.9, 0.7, 0.6, 0.1],   # P(Rain = No)
        [0.1, 0.3, 0.4, 0.9]    # P(Rain = Yes)
    ],
    evidence=['Cloudy', 'Humidity'],
    evidence_card=[2, 2]
)

# Step 3: Add CPDs to model
model.add_cpds(cpd_cloudy, cpd_humidity, cpd_rain)

# Step 4: Check model validity
print("Model Valid:", model.check_model())

# Step 5: Perform inference
infer = VariableElimination(model)
result = infer.query(variables=['Rain'], evidence={'Cloudy': 1})

print("\nProbability of Rain given Cloudy = Yes:")
print(result)

# Step 6: Draw Bayesian Network Graph
G = nx.DiGraph()
G.add_edges_from([('Cloudy', 'Rain'), ('Humidity', 'Rain')])

nx.draw(G, with_labels=True)
plt.title("Bayesian Network DAG")
plt.savefig("dag.png")   # SAVE DAG IMAGE
plt.show()

# Step 7: Plot Bar Chart
labels = ['Rain', 'No Rain']
values = [0.6, 0.4]

plt.bar(labels, values)
plt.title("Rain Prediction Probability")
plt.savefig("chart.png")   # SAVE BAR CHART IMAGE
plt.show()