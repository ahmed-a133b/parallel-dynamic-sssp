#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <string>
#include <chrono>

using namespace std;

const int INF = numeric_limits<int>::max();

// Graph representation using adjacency list
struct Edge {
    int to, weight;
};
vector<vector<Edge>> adj;
vector<int> dist, parent;

// Dijkstra's algorithm for SSSP
void dijkstra(int source, int n) {
    dist.assign(n + 1, INF);
    parent.assign(n + 1, -1);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    
    dist[source] = 0;
    pq.push({0, source});
    
    int processed = 0;
    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        if (d > dist[u]) continue;
        
        for (const Edge& e : adj[u]) {
            int v = e.to, w = e.weight;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
        processed++;
        if (processed % 10000 == 0) {
            cout << "[DEBUG] Processed " << processed << " vertices in Dijkstra's" << endl;
        }
        if (processed > n) {
            cerr << "[ERROR] Dijkstra's processed too many vertices, possible infinite loop" << endl;
            break;
        }
    }
    cout << "[DEBUG] Dijkstra's completed, processed " << processed << " vertices" << endl;
}

// Read graph from DIMACS file
void read_graph(const string& filename, int& n, int& m) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "[ERROR] Cannot open graph file: " << filename << endl;
        exit(1);
    }
    string line;
    int edges_read = 0;
    while (getline(infile, line)) {
        if (line[0] == 'p') {
            sscanf(line.c_str(), "p sp %d %d", &n, &m);
            adj.resize(n + 1);
            cout << "[DEBUG] Graph size: " << n << " vertices, " << m << " edges" << endl;
        } else if (line[0] == 'a') {
            int u, v, w;
            sscanf(line.c_str(), "a %d %d %d", &u, &v, &w);
            adj[u].push_back({v, w});
            edges_read++;
            if (edges_read % 100000 == 0) {
                cout << "[DEBUG] Read " << edges_read << " edges" << endl;
            }
        }
    }
    infile.close();
    cout << "[DEBUG] Finished reading graph, total edges read: " << edges_read << endl;
}



int main() {
    int n, m;
    string graph_file = "USA-road-d.NY.gr";
    
    
    // Read graph
    auto start = chrono::high_resolution_clock::now();
    read_graph(graph_file, n, m);
    auto end = chrono::high_resolution_clock::now();
    cout << "[DEBUG] Graph reading took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    // Output initial sssp tree
    dijkstra(1, n); // Compute SSSP from source node 1
    ofstream firstoutfile("initial_distances.txt");
    if (!firstoutfile.is_open()) {
        cerr << "[ERROR] Cannot open output file: Initial_distances.txt" << endl;
        exit(1);
    }
    firstoutfile << "[DEBUG] Final distances (all reachable vertices):\n";
    int printed = 0;
    for (int i = 1; i <= n; ++i) {
        if (dist[i] != INF) {
            firstoutfile << "Distance to " << i << ": " << dist[i] << "\n";
            printed++;
        }
    }
    if (printed == 0) {
        firstoutfile << "[WARNING] No valid distances found\n";
    }
    firstoutfile << "[DEBUG] Total reachable vertices: " << printed << "\n";
    firstoutfile.close();
   
    
    return 0;
}
