#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <string>
#include <chrono>
#include <set>

using namespace std;

const int INF = numeric_limits<int>::max();

// Graph representation using adjacency list
struct Edge {
    int to, weight;
};
vector<vector<Edge>> adj;
vector<int> dist, parent;
vector<int> prev_dist; // To store distances before edge change
set<int> affected_vertices; // To track vertices with changed distances

// Dijkstra's algorithm for SSSP
void dijkstra(int source, int n) {
    prev_dist = dist; // Save previous distances
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
                // Check if distance changed from previous run
                if (prev_dist[v] != dist[v]) {
                    affected_vertices.insert(v);
                }
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

// Apply edge change (insertion or deletion)
void apply_edge_change(char op, int u, int v, int w) {
    if (op == 'I') {
        adj[u].push_back({v, w});
        adj[v].push_back({u, w}); // Undirected graph
        cout << "[DEBUG] Inserted edge: " << u << " <-> " << v << ", weight: " << w << endl;
    } else if (op == 'D') {
        bool found = false;
        for (auto it = adj[u].begin(); it != adj[u].end(); ++it) {
            if (it->to == v) {
                adj[u].erase(it);
                found = true;
                break;
            }
        }
        for (auto it = adj[v].begin(); it != adj[v].end(); ++it) {
            if (it->to == u) {
                adj[v].erase(it);
                break;
            }
        }
        if (found) {
            cout << "[DEBUG] Deleted edge: " << u << " <-> " << v << endl;
        } else {
            cout << "[WARNING] Edge " << u << " <-> " << v << " not found for deletion" << endl;
        }
    }
}

int main() {
    int n, m;
    string graph_file = "USA-road-d.NY.gr";
    string changes_file = "edge_changes.txt";
    
    // Read graph
    auto start = chrono::high_resolution_clock::now();
    read_graph(graph_file, n, m);
    auto end = chrono::high_resolution_clock::now();
    cout << "[DEBUG] Graph reading took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    
    // Run initial Dijkstra to set baseline distances
    dist.assign(n + 1, INF);
    dijkstra(1, n);
    
    // Read and process edge changes
    ifstream changes(changes_file);
    if (!changes.is_open()) {
        cerr << "[ERROR] Cannot open changes file: " << changes_file << endl;
        exit(1);
    }
    string line;
    int change_count = 0;
    start = chrono::high_resolution_clock::now();
    while (getline(changes, line)) {
        if (line[0] == '#') continue;
        char op;
        int u, v, w;
        if (sscanf(line.c_str(), "%c %d %d %d", &op, &u, &v, &w) != 4) {
            cout << "[WARNING] Invalid change format: " << line << endl;
            continue;
        }
        if (u < 1 || u > n || v < 1 || v > n) {
            cout << "[WARNING] Invalid vertex in change: " << line << endl;
            continue;
        }
        apply_edge_change(op, u, v, w);
        dijkstra(1, n); // Recompute SSSP from source node 1
        change_count++;
        if (change_count % 1000 == 0) {
            cout << "[DEBUG] Processed " << change_count << " edge changes" << endl;
        }
    }
    changes.close();
    end = chrono::high_resolution_clock::now();
    cout << "[DEBUG] Processed " << change_count << " edge changes in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    
    // Output distances for affected vertices to file
    ofstream outfile("serial_distances.txt");
    if (!outfile.is_open()) {
        cerr << "[ERROR] Cannot open output file: serial_distances.txt" << endl;
        exit(1);
    }
    outfile << "[DEBUG] Final distances (affected vertices only):\n";
    int printed = 0;
    for (int v : affected_vertices) {
        if (dist[v] != INF) {
            outfile << "Distance to " << v << ": " << dist[v] << "\n";
            printed++;
        }
    }
    if (printed == 0) {
        outfile << "[WARNING] No affected vertices with valid distances found\n";
    }
    outfile << "[DEBUG] Total affected vertices: " << printed << "\n";
    outfile.close();
    
    return 0;
}