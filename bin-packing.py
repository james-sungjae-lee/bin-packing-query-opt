import pickle
from tqdm import tqdm
from ortools.linear_solver import pywraplp

data = pickle.load(open('./sps-01_51_50.pkl', 'rb'))
len(data)

# {instance: {region: count, region: count ...} ... } 형태의 dictionary build
region_cnt_dict = {}

for query in data:
    instance_type = query[1]
    results = query[3]

    if instance_type not in region_cnt_dict:
        region_cnt_dict[instance_type] = {}
    
    for score in results:
        region = score['Region']
        az = score['AvailabilityZoneId']
        
        if region not in region_cnt_dict[instance_type]:
            region_cnt_dict[instance_type][region] = 1
        else:
            region_cnt_dict[instance_type][region] += 1
            
# {instance: [(region, count), (region, count) ...], ...} 형태로 변경

region_cnt = {}

for instance, query in region_cnt_dict.items():
    if instance not in region_cnt:
        region_cnt[instance] = []
        region_cnt[instance].extend([(key, val) for key, val in query.items()])
    else:
        region_cnt[instance].extend([(key, val) for key, val in query.items()])
        
        
# bin packing algorithm

def create_data_model(weights, capacity):
    """Create the data for the example."""
    data = {}
    data['weights'] = weights
    data['items'] = list(range(len(weights)))
    data['bins'] = data['items']
    data['bin_capacity'] = capacity
    return data

def bin_packing(weights, capacity):
    bin_index_list = []
    data = create_data_model(weights, capacity)
    solver = pywraplp.Solver.CreateSolver('CBC')
    x = {}
    for i in data['items']:
        for j in data['bins']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
            
    y = {}
    for j in data['bins']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)
        
    for i in data['items']:
        solver.Add(sum(x[i, j] for j in data['bins']) == 1)

    for j in data['bins']:
        solver.Add(
            sum(x[(i, j)] * data['weights'][i] for i in data['items']) <= y[j] *
            data['bin_capacity'])

    solver.Minimize(solver.Sum([y[j] for j in data['bins']]))
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        num_bins = 0.
        for j in data['bins']:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_weight = 0
                for i in data['items']:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += data['weights'][i]
                if bin_weight > 0:
                    num_bins += 1
                    bin_index_list.append((bin_items, bin_weight))
        return bin_index_list
    else:
        print('The problem does not have an optimal solution.')
        
workloads = []

for instance, query in tqdm(region_cnt.items()):
    weights = [weight for region, weight in query]
    bin_index_list = bin_packing(weights, 10)
    
    for bin_index, bin_weight in bin_index_list:
        bin_regions = [query[x][0] for x in bin_index]
        workloads.append([instance, bin_regions, bin_weight])
        
print(len([x[2] for x in workloads]))
