import pickle
import numpy as np
from collections import deque

from ..utils import dataloader as loader

def bfs(start, visited, n, adjacency_list):
    cluster = []
    Q = deque()
    Q.append(start)
    this_visited = set()
    this_visited.add(start)
    while Q:
        curr = Q.popleft()
        cluster.append(curr)
        visited.add(curr)
        if len(cluster) >= n:
            break
        for adj in adjacency_list.get(curr, []):
            if adj in this_visited or adj in visited:
                continue
            Q.append(adj)
            this_visited.add(adj)
    return cluster

def fips_to_index(fips, info=None):
    if info is None:
        info = loader.load_info_raw(fips_info=True)
    try:
        return info[info['FIPS'] == fips].index[0]
    except IndexError:
        return None

def to_indices(cluster, info=None):
    indices = []
    for fips in cluster:
        i = fips_to_index(fips, info)
        if i is not None:
            indices.append(i)
    return indices

def cluster_counties(k=4, min_size=4, return_indices=True, save_file=True):
    '''
    k clusters per state
    min_size - min cluster size
    '''
    info = loader.load_info_raw(fips_info=True)
    data = loader.load_info_raw()
    deaths = data.iloc[:, [1,-1]]
    adjacency_list = loader.load_instate_adjacency_list()

    all_states = info['State'].unique()

    states_to_fips = {}

    for state in all_states:
        fips = info[info['State'] == state]['FIPS'].to_list()
        states_to_fips[state] = fips

    for node in adjacency_list:
        adj = adjacency_list[node]
        try:
            adj = sorted(adj, key=lambda fips: deaths[
                deaths['FIPS'] == fips].iloc[:, -1].to_list()[0], reverse=True)
        except IndexError:
            pass
        adjacency_list[node] = adj

    clusters = []
    cluster_id = {}
    count = 0
    for state in states_to_fips:
        counties = states_to_fips[state]
        sorted_counties = sorted(counties, key=lambda fips: deaths[
            deaths['FIPS'] == fips].iloc[:, -1].to_list()[0], reverse=True)
        
        s = len(counties) // k
        size = max(s, min_size)
        
        visited = set()
        for county in sorted_counties:
            remaining = len(counties) - len(visited)
            if 2*size > remaining:
                size = remaining
            if county not in visited:
                cluster = bfs(county, visited, size, adjacency_list)
                clusters.append(cluster)
                for c in cluster:
                    cluster_id[c] = count
                count += 1

    if return_indices:
        info = loader.load_info_raw(fips_info=True)
        clusters = list(map(lambda clu: to_indices(clu, info), clusters))
        new_cluster_id = {}
        for fips in cluster_id:
            i = cluster_id[fips]
            new_cluster_id[fips_to_index(fips, info)] = i
        cluster_id = new_cluster_id

    if save_file:
        loader.save_to_otherdata(clusters, 'clusters.dat')
        loader.save_to_otherdata(cluster_id, 'cluster_id.dat')

    return clusters, cluster_id