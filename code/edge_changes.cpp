#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <string>

using namespace std;

// Structure to represent an edge
struct Edge {
    int u, v, weight;
};

// Read graph from DIMACS file and collect edges
void read_graph(const string& filename, int& n, int& m, vector<Edge>& edges, set<pair<int, int>>& edge_set) {
    ifstream infile(filename);
    string line;
    while (getline(infile, line)) {
        if (line[0] == 'p') {
            sscanf(line.c_str(), "p sp %d %d", &n, &m);
        } else if (line[0] == 'a') {
            int u, v, w;
            sscanf(line.c_str(), "a %d %d %d", &u, &v, &w);
            edges.push_back({u, v, w});
            edge_set.insert({min(u, v), max(u, v)}); // Store undirected edge
        }
    }
    infile.close();
}

// Generate edge changes
void generate_edge_changes(const vector<Edge>& edges, set<pair<int, int>>& edge_set, int n, int num_changes, const string& output_file) {
    int num_deletions = num_changes / 2; // 5,000 deletions
    int num_insertions = num_changes - num_deletions; // 5,000 insertions
    
    // Random number generator
    random_device rd;
    mt19937 gen(42); // Fixed seed for reproducibility
    uniform_int_distribution<> vertex_dist(1, n);
    uniform_int_distribution<> weight_dist(100, 2000); // Weight range based on sample
    uniform_int_distribution<> edge_index_dist(0, edges.size() - 1);
    
    vector<string> changes;
    
    // Generate deletions
    set<int> selected_indices; // To avoid duplicate deletions
    while (selected_indices.size() < num_deletions) {
        int idx = edge_index_dist(gen);
        if (selected_indices.find(idx) == selected_indices.end()) {
            selected_indices.insert(idx);
            const Edge& e = edges[idx];
            changes.push_back("D " + to_string(e.u) + " " + to_string(e.v) + " " + to_string(e.weight));
            edge_set.erase({min(e.u, e.v), max(e.u, e.v)}); // Remove from edge set
        }
    }
    
    // Generate insertions
    for (int i = 0; i < num_insertions; ++i) {
        int u, v, w;
        pair<int, int> edge;
        do {
            u = vertex_dist(gen);
            v = vertex_dist(gen);
            if (u > v) swap(u, v); // Ensure u <= v
            edge = {u, v};
        } while (u == v || edge_set.find(edge) != edge_set.end()); // Avoid self-loops and existing edges
        w = weight_dist(gen);
        changes.push_back("I " + to_string(u) + " " + to_string(v) + " " + to_string(w));
        edge_set.insert(edge); // Add to edge set
    }
    
    // Shuffle changes to mix deletions and insertions
    shuffle(changes.begin(), changes.end(), gen);
    
    // Write to output file
    ofstream outfile(output_file);
    outfile << "# Edge changes for USA-road-d.NY (format: <operation> <u> <v> <weight>)\n";
    outfile << "# Operations: D (deletion), I (insertion)\n";
    for (const string& change : changes) {
        outfile << change << "\n";
    }
    outfile.close();
}

int main() {
    int n, m;
    string graph_file = "USA-road-d.NY.gr";
    string output_file = "edge_changes.txt";
    
    vector<Edge> edges;
    set<pair<int, int>> edge_set; // To track existing edges (undirected)
    int changes = 500;
    // Read graph
    read_graph(graph_file, n, m, edges, edge_set);
    
    // Generate edge changes
    generate_edge_changes(edges, edge_set, n, changes, output_file);
    cout << "Generated " << changes << " edge changes in " << output_file << endl;
    
    
    return 0;
}