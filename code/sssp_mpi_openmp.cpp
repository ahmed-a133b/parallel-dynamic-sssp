#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <openmpi/mpi.h>
#include <omp.h>
#include <metis.h>
#include <stdbool.h>
#include <time.h>

#define INF INT_MAX
#define MAX_LINE 256

// Structure for an edge
typedef struct {
    int to, weight;
} Edge;

// Structure for a graph partition
typedef struct {
    int n_local;          // Number of local vertices
    int* local_vertices;  // Mapping of local vertex IDs to global IDs
    int** adj;            // Adjacency list: adj[i] is array of edges for local vertex i
    int* adj_sizes;       // Number of edges per local vertex
    int* dist;            // Distance from source for local vertices
    int* parent;          // Parent in SSSP tree for local vertices
    bool* affected;       // Whether vertex is affected by changes
    bool* affected_del;   // Whether vertex is affected by deletion
    int n_ghost;          // Number of ghost vertices
    int* ghost_vertices;  // Global IDs of ghost vertices
    int* ghost_dist;      // Distances for ghost vertices
    int* ghost_parent;    // Parents for ghost vertices
    int* ghost_to_local;  // Mapping from global vertex ID to local/ghost index
    int* vertex_owner;    // Mapping from global vertex ID to owning rank
} Partition;

// Global variables
int rank, n_ranks;
int n_global, m_global; // Global number of vertices and edges
int source = 1;         // Source vertex for SSSP
Partition* part;        // Local partition data
MPI_Comm comm = MPI_COMM_WORLD;

// Read graph and partition with METIS
void read_graph_and_partition(const char* filename, idx_t** xadj, idx_t** adjncy) {
    n_global = 0;
    m_global = 0;
    int max_vertex = 0;
    int* edge_count = NULL;

    if (rank == 0) {
        FILE* file = fopen(filename, "r");
        if (!file) {
            fprintf(stderr, "[ERROR][Rank %d] Cannot open graph file: %s\n", rank, filename);
            MPI_Abort(comm, 1);
        }

        char line[MAX_LINE];
        // First pass: determine n_global, m_global, and count edges
        while (fgets(line, MAX_LINE, file)) {
            if (line[0] == 'p') {
                sscanf(line, "p sp %d %d", &n_global, &m_global);
                edge_count = (int*)calloc(n_global + 1, sizeof(int));
                if (!edge_count) {
                    fprintf(stderr, "[ERROR][Rank %d] Memory allocation failed for edge_count\n", rank);
                    MPI_Abort(comm, 1);
                }
            } else if (line[0] == 'a') {
                int u, v, w;
                if (sscanf(line, "a %d %d %d", &u, &v, &w) != 3) {
                    fprintf(stderr, "[ERROR][Rank %d] Invalid edge format: %s\n", rank, line);
                    MPI_Abort(comm, 1);
                }
                if (u < 1 || v < 1) {
                    fprintf(stderr, "[ERROR][Rank %d] Invalid vertex: u=%d, v=%d\n", rank, u, v);
                    MPI_Abort(comm, 1);
                }
                edge_count[u]++;
                edge_count[v]++; // Undirected graph
                max_vertex = u > max_vertex ? u : max_vertex;
                max_vertex = v > max_vertex ? v : max_vertex;
            }
        }
        fclose(file);

        if (max_vertex > n_global) {
            fprintf(stderr, "[ERROR][Rank %d] Vertex %d exceeds declared n_global %d\n", rank, max_vertex, n_global);
            MPI_Abort(comm, 1);
        }
        printf("[DEBUG] Graph size: %d vertices, %d edges\n", n_global, m_global);
    }

    // Broadcast n_global and m_global
    MPI_Bcast(&n_global, 1, MPI_INT, 0, comm);
    MPI_Bcast(&m_global, 1, MPI_INT, 0, comm);

    // Allocate edge_count on other ranks
    if (rank != 0) {
        edge_count = (int*)calloc(n_global + 1, sizeof(int));
        if (!edge_count) {
            fprintf(stderr, "[ERROR][Rank %d] Memory allocation failed for edge_count\n", rank);
            MPI_Abort(comm, 1);
        }
    }

    // Allocate xadj and adjncy
    *xadj = (idx_t*)calloc(n_global + 1, sizeof(idx_t));
    *adjncy = (idx_t*)calloc(4 * m_global, sizeof(idx_t)); // 2*m_global vertices + 2*m_global weights
    if (!*xadj || !*adjncy) {
        fprintf(stderr, "[ERROR][Rank %d] Memory allocation failed for xadj or adjncy\n", rank);
        MPI_Abort(comm, 1);
    }

    if (rank == 0) {
        // Compute xadj
        (*xadj)[0] = 0;
        for (int i = 1; i <= n_global; i++) {
            (*xadj)[i] = (*xadj)[i - 1] + edge_count[i];
        }

        // Validate xadj
        if ((*xadj)[n_global] != 2 * m_global) {
            fprintf(stderr, "[ERROR][Rank %d] xadj[%d] = %d, expected %d (2*m_global)\n",
                    rank, n_global, (*xadj)[n_global], 2 * m_global);
            MPI_Abort(comm, 1);
        }

        // Second pass: populate adjncy
        FILE* file = fopen(filename, "r");
        if (!file) {
            fprintf(stderr, "[ERROR][Rank %d] Cannot open graph file: %s\n", rank, filename);
            MPI_Abort(comm, 1);
        }
        char line[MAX_LINE];
        int* current_edge = (int*)calloc(n_global + 1, sizeof(int));
        while (fgets(line, MAX_LINE, file)) {
            if (line[0] == 'a') {
                int u, v, w;
                sscanf(line, "a %d %d %d", &u, &v, &w);
                int idx_u = (*xadj)[u - 1] + current_edge[u]++;
                int idx_v = (*xadj)[v - 1] + current_edge[v]++;
                (*adjncy)[idx_u] = v - 1; // 0-based for METIS
                (*adjncy)[idx_v] = u - 1;
                (*adjncy)[idx_u + 2 * m_global] = w;
                (*adjncy)[idx_v + 2 * m_global] = w;
            }
        }
        fclose(file);
        free(current_edge);

        // Validate adjncy
        for (int i = 0; i < 2 * m_global; i++) {
            if ((*adjncy)[i] < 0 || (*adjncy)[i] >= n_global) {
                fprintf(stderr, "[ERROR][Rank %d] Invalid vertex in adjncy[%d] = %d\n",
                        rank, i, (*adjncy)[i]);
                MPI_Abort(comm, 1);
            }
        }
        printf("[DEBUG] xadj[%d] = %d, adjncy size = %d\n", n_global, (*xadj)[n_global], 4 * m_global);
    }

    // Broadcast xadj and adjncy to all ranks
    MPI_Bcast(*xadj, n_global + 1, MPI_INT, 0, comm);
    MPI_Bcast(*adjncy, 4 * m_global, MPI_INT, 0, comm);

    free(edge_count);
}

// Initialize partition data after METIS partitioning
void initialize_partition(idx_t* part_map, idx_t* xadj, idx_t* adjncy) {
    // Count local and ghost vertices
    int* local_count = (int*)calloc(n_ranks, sizeof(int));
    for (int i = 0; i < n_global; i++) {
        if (part_map[i] == rank) {
            local_count[rank]++;
        }
    }

    part = (Partition*)malloc(sizeof(Partition));
    part->n_local = local_count[rank];
    part->local_vertices = (int*)malloc(part->n_local * sizeof(int));
    int local_idx = 0;
    for (int i = 0; i < n_global; i++) {
        if (part_map[i] == rank) {
            part->local_vertices[local_idx++] = i + 1; // 1-based indexing
        }
    }

    // Initialize vertex owner array
    part->vertex_owner = (int*)malloc(n_global * sizeof(int));
    for (int i = 0; i < n_global; i++) {
        part->vertex_owner[i] = part_map[i];
    }

    // Check if source vertex is local or ghost
    int source_owner = -1;
    if (source > 0 && source <= n_global) {
        source_owner = part_map[source - 1];
        if (rank == 0) {
            printf("[DEBUG] Source vertex %d is owned by rank %d\n", source, source_owner);
        }
    }

    // Identify ghost vertices
    int* ghost_set = (int*)calloc(n_global, sizeof(int));
    for (int i = 0; i < part->n_local; i++) {
        int u_global = part->local_vertices[i] - 1; // 0-based for METIS
        for (idx_t j = xadj[u_global]; j < xadj[u_global + 1]; j++) {
            int v_global = adjncy[j];
            if (part_map[v_global] != rank) {
                ghost_set[v_global] = 1;
            }
        }
    }

    // If I'm not the owner of the source, ensure it's in my ghost set
    if (source_owner != rank && source > 0 && source <= n_global) {
        ghost_set[source - 1] = 1;
        if (rank == 0) {
            printf("[DEBUG] Adding source vertex %d as ghost for rank %d\n", source, rank);
        }
    }

    part->n_ghost = 0;
    for (int i = 0; i < n_global; i++) {
        if (ghost_set[i]) part->n_ghost++;
    }
    part->ghost_vertices = (int*)malloc(part->n_ghost * sizeof(int));
    int ghost_idx = 0;
    for (int i = 0; i < n_global; i++) {
        if (ghost_set[i]) {
            part->ghost_vertices[ghost_idx++] = i + 1; // 1-based
        }
    }

    // Initialize adjacency list
    part->adj = (int**)malloc(part->n_local * sizeof(int*));
    part->adj_sizes = (int*)calloc(part->n_local, sizeof(int));
    for (int i = 0; i < part->n_local; i++) {
        int u_global = part->local_vertices[i] - 1;
        part->adj_sizes[i] = xadj[u_global + 1] - xadj[u_global];
        part->adj[i] = (int*)malloc(2 * part->adj_sizes[i] * sizeof(int));
        for (idx_t j = xadj[u_global], k = 0; j < xadj[u_global + 1]; j++, k += 2) {
            part->adj[i][k] = adjncy[j] + 1; // to
            part->adj[i][k + 1] = adjncy[j + 2 * m_global]; // weight
        }
    }

    // Initialize distance, parent, and affected arrays
    part->dist = (int*)malloc(part->n_local * sizeof(int));
    part->parent = (int*)malloc(part->n_local * sizeof(int));
    part->affected = (bool*)calloc(part->n_local, sizeof(bool));
    part->affected_del = (bool*)calloc(part->n_local, sizeof(bool));
    part->ghost_dist = (int*)malloc(part->n_ghost * sizeof(int));
    part->ghost_parent = (int*)malloc(part->n_ghost * sizeof(int));
    
    // Initialize ghost_to_local mapping
    part->ghost_to_local = (int*)malloc(n_global * sizeof(int));
    for (int i = 0; i < n_global; i++) {
        part->ghost_to_local[i] = -1; // Default: not known to this process
    }
    
    // Map local vertices to their indices
    for (int i = 0; i < part->n_local; i++) {
        if (part->local_vertices[i] > 0 && part->local_vertices[i] <= n_global) {
            part->ghost_to_local[part->local_vertices[i] - 1] = i;
        } else {
            fprintf(stderr, "[ERROR][Rank %d] Invalid local vertex ID: %d\n", rank, part->local_vertices[i]);
            MPI_Abort(comm, 1);
        }
    }
    
    // Map ghost vertices to their indices
    for (int i = 0; i < part->n_ghost; i++) {
        if (part->ghost_vertices[i] > 0 && part->ghost_vertices[i] <= n_global) {
            part->ghost_to_local[part->ghost_vertices[i] - 1] = i + part->n_local;
        } else {
            fprintf(stderr, "[ERROR][Rank %d] Invalid ghost vertex ID: %d\n", rank, part->ghost_vertices[i]);
            MPI_Abort(comm, 1);
        }
    }
    
    // Debug verification of ghost_to_local mapping
    if (rank == 0) {
        printf("[DEBUG] Checking ghost_to_local mappings...\n");
        int errors = 0;
        for (int i = 0; i < part->n_local; i++) {
            int global_id = part->local_vertices[i] - 1;
            if (global_id < 0 || global_id >= n_global) {
                printf("[ERROR] Invalid local vertex global ID: %d\n", global_id + 1);
                errors++;
                continue;
            }
            if (part->ghost_to_local[global_id] != i) {
                printf("[ERROR] Incorrect local mapping for vertex %d: expected %d, got %d\n",
                       global_id + 1, i, part->ghost_to_local[global_id]);
                errors++;
            }
        }
        
        for (int i = 0; i < part->n_ghost; i++) {
            int global_id = part->ghost_vertices[i] - 1;
            if (global_id < 0 || global_id >= n_global) {
                printf("[ERROR] Invalid ghost vertex global ID: %d\n", global_id + 1);
                errors++;
                continue;
            }
            if (part->ghost_to_local[global_id] != i + part->n_local) {
                printf("[ERROR] Incorrect ghost mapping for vertex %d: expected %d, got %d\n",
                       global_id + 1, i + part->n_local, part->ghost_to_local[global_id]);
                errors++;
            }
        }
        
        if (errors == 0) {
            printf("[DEBUG] Ghost-to-local mappings verified successfully\n");
        } else {
            printf("[WARNING] Found %d errors in ghost-to-local mappings\n", errors);
        }
    }

    // Verify source vertex is properly mapped
    int source_idx = -1;
    if (source > 0 && source <= n_global) {
        source_idx = part->ghost_to_local[source - 1];
        if (source_idx >= 0) {
            if (source_idx < part->n_local) {
                if (rank == 0) {
                    printf("[DEBUG] Source vertex %d is mapped to local index %d in rank %d\n", 
                           source, source_idx, rank);
                }
            } else if (source_idx < part->n_local + part->n_ghost) {
                if (rank == 0) {
                    printf("[DEBUG] Source vertex %d is mapped to ghost index %d in rank %d\n", 
                           source, source_idx - part->n_local, rank);
                }
            }
        } else if (rank == 0) {
            printf("[WARNING] Source vertex %d is not mapped in rank %d\n", source, rank);
        }
    }

    free(local_count);
    free(ghost_set);
}

// Add this function after initialize_partition to debug the graph structure
void debug_source_vertex() {
    // Find the source vertex in the graph
    int source_owner = -1;
    if (source > 0 && source <= n_global) {
        source_owner = part->vertex_owner[source - 1];
    }
    
    MPI_Barrier(comm);
    
    // Each rank reports if they own the source vertex
    if (rank == source_owner) {
        printf("[DEBUG][Rank %d] I am the owner of source vertex %d\n", rank, source);
        
        // Find the local index of source vertex
        int source_idx = -1;
        for (int i = 0; i < part->n_local; i++) {
            if (part->local_vertices[i] == source) {
                source_idx = i;
                break;
            }
        }
        
        if (source_idx >= 0) {
            printf("[DEBUG][Rank %d] Source vertex %d has local index %d\n", 
                   rank, source, source_idx);
            
            // Print number of neighbors
            printf("[DEBUG][Rank %d] Source vertex has %d neighbors:\n", 
                   rank, part->adj_sizes[source_idx]);
            
            // Print all neighbors of source
            for (int i = 0; i < part->adj_sizes[source_idx]; i++) {
                int neighbor = part->adj[source_idx][2 * i];
                int weight = part->adj[source_idx][2 * i + 1];
                int owner = -1;
                if (neighbor - 1 >= 0 && neighbor - 1 < n_global) {
                    owner = part->vertex_owner[neighbor - 1];
                }
                printf("[DEBUG][Rank %d] Source neighbor: %d, weight: %d, owner: %d\n", 
                       rank, neighbor, weight, owner);
            }
        } else {
            printf("[ERROR][Rank %d] Could not find source vertex %d in local vertices!\n", 
                   rank, source);
        }
    }
    
    // Each rank checks if they have the source as a ghost vertex
    int ghost_idx = -1;
    for (int i = 0; i < part->n_ghost; i++) {
        if (part->ghost_vertices[i] == source) {
            ghost_idx = i;
            break;
        }
    }
    
    if (ghost_idx >= 0) {
        printf("[DEBUG][Rank %d] I have source vertex %d as ghost with index %d\n", 
               rank, source, ghost_idx);
    }
    
    MPI_Barrier(comm);
}

// Improved read_initial_sssp function to correctly load initial distances
void read_initial_sssp(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "[ERROR][Rank %d] Cannot open initial SSSP file: %s\n", rank, filename);
        MPI_Abort(comm, 1);
    }

    if (rank == 0) {
        printf("[INFO] Reading initial distances from %s\n", filename);
    }

    // Initialize all distances to INF
    for (int i = 0; i < part->n_local; i++) {
        part->dist[i] = INF;
        part->parent[i] = -1;
        part->affected[i] = false;
    }
    for (int i = 0; i < part->n_ghost; i++) {
        part->ghost_dist[i] = INF;
        part->ghost_parent[i] = -1;
    }

    char line[MAX_LINE];
    int loaded_count = 0;
    
    // Skip the first line which is a comment
    fgets(line, MAX_LINE, file);
    
    while (fgets(line, MAX_LINE, file)) {
        int v, d;
        if (sscanf(line, "Distance to %d: %d", &v, &d) == 2) {
            if (v <= 0 || v > n_global) {
                if (rank == 0) {
                    printf("[WARNING] Invalid vertex ID in initial distances: %d\n", v);
                }
                continue;
            }
            
            // Look up the vertex in our local data structures
            int idx = -1;
            if (v - 1 >= 0 && v - 1 < n_global) {
                idx = part->ghost_to_local[v - 1];
            }
            
            if (idx >= 0 && idx < part->n_local) {
                // This is a local vertex
                part->dist[idx] = d;
                part->parent[idx] = -1; // Root has no parent
                part->affected[idx] = true;  // Mark as affected for updates
                loaded_count++;
                
                if (rank == 0 && loaded_count <= 5) {
                    printf("[DEBUG] Loaded initial distance for local vertex %d: %d\n", v, d);
                }
            } 
            else if (idx >= part->n_local && idx < part->n_local + part->n_ghost) {
                // This is a ghost vertex
                int ghost_idx = idx - part->n_local;
                if (ghost_idx >= 0 && ghost_idx < part->n_ghost) {
                    part->ghost_dist[ghost_idx] = d;
                    part->ghost_parent[ghost_idx] = -1;
                    loaded_count++;
                    
                    if (rank == 0 && loaded_count <= 5) {
                        printf("[DEBUG] Loaded initial distance for ghost vertex %d: %d\n", v, d);
                    }
                }
            }
        }
    }
    fclose(file);

    // Ensure source vertex is consistent and properly initialized
    int source_idx = -1;
    if (source - 1 >= 0 && source - 1 < n_global) {
        source_idx = part->ghost_to_local[source - 1];
    }
    
    if (source_idx >= 0 && source_idx < part->n_local) {
        // Source is a local vertex for this process
        part->dist[source_idx] = 0;
        part->parent[source_idx] = -1;
        part->affected[source_idx] = true;
        
        if (rank == 0) {
            printf("[DEBUG] Source vertex %d is local to rank %d, setting distance to 0\n", source, rank);
        }
        
        // Also initialize direct neighbors of source
        for (int j = 0; j < part->adj_sizes[source_idx]; j++) {
            int v_global = part->adj[source_idx][2 * j];
            int w = part->adj[source_idx][2 * j + 1];
            
            if (v_global <= 0 || v_global > n_global) continue;
            
            int v_idx = part->ghost_to_local[v_global - 1];
            
            if (v_idx >= 0 && v_idx < part->n_local) {
                // This is a local vertex
                if (part->dist[v_idx] > w) {
                    part->dist[v_idx] = w;
                    part->parent[v_idx] = source;
                    part->affected[v_idx] = true;
                    
                    if (rank == 0) {
                        printf("[DEBUG] Initialized neighbor %d of source with distance %d\n", 
                               v_global, w);
                    }
                }
            } 
            else if (v_idx >= part->n_local && v_idx < part->n_local + part->n_ghost) {
                // This is a ghost vertex
                int ghost_idx = v_idx - part->n_local;
                if (ghost_idx >= 0 && ghost_idx < part->n_ghost) {
                    if (part->ghost_dist[ghost_idx] > w) {
                        part->ghost_dist[ghost_idx] = w;
                        part->ghost_parent[ghost_idx] = source;
                        
                        if (rank == 0) {
                            printf("[DEBUG] Initialized ghost neighbor %d of source with distance %d\n", 
                                  v_global, w);
                        }
                    }
                }
            }
        }
    } 
    else if (source_idx >= part->n_local && source_idx < part->n_local + part->n_ghost) {
        // Source is a ghost vertex for this process
        int ghost_idx = source_idx - part->n_local;
        if (ghost_idx >= 0 && ghost_idx < part->n_ghost) {
            part->ghost_dist[ghost_idx] = 0;
            part->ghost_parent[ghost_idx] = -1;
            
            if (rank == 0) {
                printf("[DEBUG] Source vertex %d is a ghost vertex for rank %d, setting distance to 0\n", source, rank);
            }
            
            // Check if any local vertices are connected to the source
            for (int i = 0; i < part->n_local; i++) {
                for (int j = 0; j < part->adj_sizes[i]; j++) {
                    if (part->adj[i][2 * j] == source) {
                        int w = part->adj[i][2 * j + 1];
                        if (part->dist[i] > w) {
                            part->dist[i] = w;
                            part->parent[i] = source;
                            part->affected[i] = true;
                            
                            if (rank == 0) {
                                printf("[DEBUG] Initialized vertex %d connected to source with distance %d\n", 
                                       part->local_vertices[i], w);
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Source is not in this process's vertices at all
        if (rank == 0) {
            printf("[DEBUG] Source vertex %d is not known to rank %d\n", source, rank);
        }
    }
    
    // Print the total number of distances loaded
    int total_loaded;
    MPI_Reduce(&loaded_count, &total_loaded, 1, MPI_INT, MPI_SUM, 0, comm);
    
    if (rank == 0) {
        printf("[INFO] Loaded %d initial distances (%d in rank 0)\n", total_loaded, loaded_count);
    }
    
    MPI_Barrier(comm);
}

// Find edge between u and v in u's adjacency list
int find_edge(int u_local_idx, int v_global) {
    for (int i = 0; i < part->adj_sizes[u_local_idx]; i++) {
        if (part->adj[u_local_idx][2 * i] == v_global) {
            return i;
        }
    }
    return -1;
}

// Apply edge change
void apply_edge_change(char op, int u, int v, int w) {
    // Safety check for vertex IDs
    if (u <= 0 || u > n_global || v <= 0 || v > n_global) {
        if (rank == 0) {
            printf("[WARNING] Invalid vertex in edge change: %c %d %d %d\n", op, u, v, w);
        }
        return;
    }

    int u_idx = part->ghost_to_local[u - 1];
    int v_idx = part->ghost_to_local[v - 1];
    
    // Skip if both vertices are unknown to this process
    if (u_idx < 0 && v_idx < 0) {
        return;
    }

    if (rank == 0) {
        printf("[DEBUG] Processing edge change: %c %d %d %d (u_idx=%d, v_idx=%d)\n", 
               op, u, v, w, u_idx, v_idx);
    }

    if (op == 'I') { // Insert edge
        // Process u if it's a local vertex
        if (u_idx >= 0 && u_idx < part->n_local) {
            // Check if edge already exists
            int edge_idx = find_edge(u_idx, v);
            if (edge_idx >= 0) {
                // Update weight if edge exists
                part->adj[u_idx][2 * edge_idx + 1] = w;
            } else {
                // Add new edge
                int new_size = part->adj_sizes[u_idx] + 1;
                part->adj[u_idx] = (int*)realloc(part->adj[u_idx], 2 * new_size * sizeof(int));
                if (!part->adj[u_idx]) {
                    fprintf(stderr, "[ERROR][Rank %d] Memory reallocation failed\n", rank);
                    MPI_Abort(comm, 1);
                }
                part->adj[u_idx][2 * part->adj_sizes[u_idx]] = v;
                part->adj[u_idx][2 * part->adj_sizes[u_idx] + 1] = w;
                part->adj_sizes[u_idx]++;
            }
        }
        
        // Process v if it's a local vertex
        if (v_idx >= 0 && v_idx < part->n_local) {
            // Check if edge already exists
            int edge_idx = find_edge(v_idx, u);
            if (edge_idx >= 0) {
                // Update weight if edge exists
                part->adj[v_idx][2 * edge_idx + 1] = w;
            } else {
                // Add new edge
                int new_size = part->adj_sizes[v_idx] + 1;
                part->adj[v_idx] = (int*)realloc(part->adj[v_idx], 2 * new_size * sizeof(int));
                if (!part->adj[v_idx]) {
                    fprintf(stderr, "[ERROR][Rank %d] Memory reallocation failed\n", rank);
                    MPI_Abort(comm, 1);
                }
                part->adj[v_idx][2 * part->adj_sizes[v_idx]] = u;
                part->adj[v_idx][2 * part->adj_sizes[v_idx] + 1] = w;
                part->adj_sizes[v_idx]++;
            }
        }
        
        if (rank == 0) {
            printf("[DEBUG] Inserted edge: %d <-> %d, weight: %d\n", u, v, w);
        }
    } else if (op == 'D') { // Delete edge
        // Process u if it's a local vertex
        if (u_idx >= 0 && u_idx < part->n_local) {
            int edge_idx = find_edge(u_idx, v);
            if (edge_idx >= 0) {
                // Remove edge by shifting remaining edges
                for (int i = edge_idx; i < part->adj_sizes[u_idx] - 1; i++) {
                    part->adj[u_idx][2 * i] = part->adj[u_idx][2 * (i + 1)];
                    part->adj[u_idx][2 * i + 1] = part->adj[u_idx][2 * (i + 1) + 1];
                }
                part->adj_sizes[u_idx]--;
                // Resize the adjacency list
                if (part->adj_sizes[u_idx] > 0) {
                    part->adj[u_idx] = (int*)realloc(part->adj[u_idx], 2 * part->adj_sizes[u_idx] * sizeof(int));
                    if (!part->adj[u_idx]) {
                        fprintf(stderr, "[ERROR][Rank %d] Memory reallocation failed\n", rank);
                        MPI_Abort(comm, 1);
                    }
                }
            }
        }
        
        // Process v if it's a local vertex
        if (v_idx >= 0 && v_idx < part->n_local) {
            int edge_idx = find_edge(v_idx, u);
            if (edge_idx >= 0) {
                // Remove edge by shifting remaining edges
                for (int i = edge_idx; i < part->adj_sizes[v_idx] - 1; i++) {
                    part->adj[v_idx][2 * i] = part->adj[v_idx][2 * (i + 1)];
                    part->adj[v_idx][2 * i + 1] = part->adj[v_idx][2 * (i + 1) + 1];
                }
                part->adj_sizes[v_idx]--;
                // Resize the adjacency list
                if (part->adj_sizes[v_idx] > 0) {
                    part->adj[v_idx] = (int*)realloc(part->adj[v_idx], 2 * part->adj_sizes[v_idx] * sizeof(int));
                    if (!part->adj[v_idx]) {
                        fprintf(stderr, "[ERROR][Rank %d] Memory reallocation failed\n", rank);
                        MPI_Abort(comm, 1);
                    }
                }
            }
        }
        
        if (rank == 0) {
            printf("[DEBUG] Deleted edge: %d <-> %d\n", u, v);
        }
    }
}

// Get vertex distance regardless of whether it's local or ghost
int get_vertex_dist(int vertex_idx) {
    if (vertex_idx < 0) {
        return INF;
    } else if (vertex_idx < part->n_local) {
        return part->dist[vertex_idx];
    } else if (vertex_idx < part->n_local + part->n_ghost) {
        return part->ghost_dist[vertex_idx - part->n_local];
    } else {
        return INF;
    }
}

// Improved process_edge_changes function for better edge change handling
void process_edge_changes(char op, int u, int v, int w) {
    // Safety check for vertex IDs
    if (u <= 0 || u > n_global || v <= 0 || v > n_global) {
        if (rank == 0) {
            printf("[WARNING] Invalid vertex in edge change: %c %d %d %d\n", op, u, v, w);
        }
        return;
    }

    // Use a safer way to get local indices
    int u_idx = -1;
    int v_idx = -1;
    
    // Find u in local or ghost vertices
    if (u - 1 >= 0 && u - 1 < n_global) {
        u_idx = part->ghost_to_local[u - 1];
    }
    
    // Find v in local or ghost vertices
    if (v - 1 >= 0 && v - 1 < n_global) {
        v_idx = part->ghost_to_local[v - 1];
    }
    
    // Skip if both vertices are unknown to this process
    if (u_idx < 0 && v_idx < 0) {
        return;
    }
    
    // Get vertex distances safely
    int dist_u = get_vertex_dist(u_idx);
    int dist_v = get_vertex_dist(v_idx);
    
    if (rank == 0) {
        printf("[DEBUG] Process edge change: %c %d %d %d (dist_u=%d, dist_v=%d)\n", 
               op, u, v, w, dist_u, dist_v);
    }
    
    if (op == 'I') { // Insert edge
        // Check if either vertex can be updated through the other
        if (dist_u < INF && (dist_v == INF || dist_v > dist_u + w)) {
            // Update v through u
            if (v_idx >= 0 && v_idx < part->n_local) {
                part->dist[v_idx] = dist_u + w;
                part->parent[v_idx] = u;
                part->affected[v_idx] = true;
                
                if (rank == 0) {
                    printf("[DEBUG] Local vertex %d updated: new dist=%d, parent=%d\n", 
                           part->local_vertices[v_idx], part->dist[v_idx], part->parent[v_idx]);
                }
            } 
            else if (v_idx >= part->n_local && v_idx < part->n_local + part->n_ghost) {
                int ghost_idx = v_idx - part->n_local;
                if (ghost_idx >= 0 && ghost_idx < part->n_ghost) {
                    part->ghost_dist[ghost_idx] = dist_u + w;
                    part->ghost_parent[ghost_idx] = u;
                    
                    if (rank == 0) {
                        printf("[DEBUG] Ghost vertex %d updated: new dist=%d, parent=%d\n", 
                               part->ghost_vertices[ghost_idx], 
                               part->ghost_dist[ghost_idx], 
                               part->ghost_parent[ghost_idx]);
                    }
                }
            }
        }
        
        if (dist_v < INF && (dist_u == INF || dist_u > dist_v + w)) {
            // Update u through v
            if (u_idx >= 0 && u_idx < part->n_local) {
                part->dist[u_idx] = dist_v + w;
                part->parent[u_idx] = v;
                part->affected[u_idx] = true;
                
                if (rank == 0) {
                    printf("[DEBUG] Local vertex %d updated: new dist=%d, parent=%d\n", 
                           part->local_vertices[u_idx], part->dist[u_idx], part->parent[u_idx]);
                }
            } 
            else if (u_idx >= part->n_local && u_idx < part->n_local + part->n_ghost) {
                int ghost_idx = u_idx - part->n_local;
                if (ghost_idx >= 0 && ghost_idx < part->n_ghost) {
                    part->ghost_dist[ghost_idx] = dist_v + w;
                    part->ghost_parent[ghost_idx] = v;
                    
                    if (rank == 0) {
                        printf("[DEBUG] Ghost vertex %d updated: new dist=%d, parent=%d\n", 
                               part->ghost_vertices[ghost_idx], 
                               part->ghost_dist[ghost_idx], 
                               part->ghost_parent[ghost_idx]);
                    }
                }
            }
        }
    } else if (op == 'D') { // Delete edge
        // Check if u's parent is v
        if (u_idx >= 0 && u_idx < part->n_local && part->parent[u_idx] == v) {
            part->dist[u_idx] = INF;  // Reset distance
            part->parent[u_idx] = -1; // Reset parent
            part->affected_del[u_idx] = true;
            part->affected[u_idx] = true;
            
            if (rank == 0) {
                printf("[DEBUG] Local vertex %d marked affected by deletion (parent was %d)\n", 
                       part->local_vertices[u_idx], v);
            }
        } 
        // Check if v's parent is u
        else if (v_idx >= 0 && v_idx < part->n_local && part->parent[v_idx] == u) {
            part->dist[v_idx] = INF;  // Reset distance
            part->parent[v_idx] = -1; // Reset parent
            part->affected_del[v_idx] = true;
            part->affected[v_idx] = true;
            
            if (rank == 0) {
                printf("[DEBUG] Local vertex %d marked affected by deletion (parent was %d)\n", 
                       part->local_vertices[v_idx], u);
            }
        }
        
        // Check ghost vertices too
        if (u_idx >= part->n_local && u_idx < part->n_local + part->n_ghost) {
            int ghost_idx = u_idx - part->n_local;
            if (ghost_idx >= 0 && ghost_idx < part->n_ghost && part->ghost_parent[ghost_idx] == v) {
                part->ghost_dist[ghost_idx] = INF;
                part->ghost_parent[ghost_idx] = -1;
                
                if (rank == 0) {
                    printf("[DEBUG] Ghost vertex %d reset by deletion\n", 
                           part->ghost_vertices[ghost_idx]);
                }
            }
        }
        
        if (v_idx >= part->n_local && v_idx < part->n_local + part->n_ghost) {
            int ghost_idx = v_idx - part->n_local;
            if (ghost_idx >= 0 && ghost_idx < part->n_ghost && part->ghost_parent[ghost_idx] == u) {
                part->ghost_dist[ghost_idx] = INF;
                part->ghost_parent[ghost_idx] = -1;
                
                if (rank == 0) {
                    printf("[DEBUG] Ghost vertex %d reset by deletion\n", 
                           part->ghost_vertices[ghost_idx]);
                }
            }
        }
        
        // Mark all affected vertices for update in the main algorithm
        if (u_idx >= 0 && u_idx < part->n_local) {
            part->affected[u_idx] = true;
        }
        if (v_idx >= 0 && v_idx < part->n_local) {
            part->affected[v_idx] = true;
        }
    }
}

// Improved synchronization function to correctly propagate distances
void sync_ghost_vertices() {
    // Create send buffer for all vertices that need to be shared
    int* send_dists = (int*)malloc(n_global * sizeof(int));
    int* send_parents = (int*)malloc(n_global * sizeof(int));
    
    // Initialize all to INF/-1
    for (int i = 0; i < n_global; i++) {
        send_dists[i] = INF;
        send_parents[i] = -1;
    }
    
    // Fill with local vertex info
    for (int i = 0; i < part->n_local; i++) {
        int global_id = part->local_vertices[i] - 1;
        if (global_id >= 0 && global_id < n_global) {
            send_dists[global_id] = part->dist[i];
            send_parents[global_id] = part->parent[i];
        }
    }
    
    // Special case: ensure source vertex has distance 0
    if (source - 1 >= 0 && source - 1 < n_global) {
        send_dists[source - 1] = 0;
    }
    
    // Allocate receive buffers
    int* recv_dists = (int*)malloc(n_global * sizeof(int));
    int* recv_parents = (int*)malloc(n_global * sizeof(int));
    
    // Initialize to INF/-1
    for (int i = 0; i < n_global; i++) {
        recv_dists[i] = INF;
        recv_parents[i] = -1;
    }
    
    // Use MPI_Allreduce to get the minimum distance for each vertex across all processes
    MPI_Allreduce(send_dists, recv_dists, n_global, MPI_INT, MPI_MIN, comm);
    
    // Special cases after reduction: ensure source vertex has distance 0
    if (source - 1 >= 0 && source - 1 < n_global) {
        recv_dists[source - 1] = 0;
    }
    
    // We need special handling for parents since we can't just take the minimum
    // First, use MPI_Allgather to collect all local distances and parents
    int* all_dists = (int*)malloc(n_ranks * n_global * sizeof(int));
    int* all_parents = (int*)malloc(n_ranks * n_global * sizeof(int));
    
    MPI_Allgather(send_dists, n_global, MPI_INT, all_dists, n_global, MPI_INT, comm);
    MPI_Allgather(send_parents, n_global, MPI_INT, all_parents, n_global, MPI_INT, comm);
    
    // For each vertex, find the rank with the minimum distance and take its parent
    for (int i = 0; i < n_global; i++) {
        int min_dist = INF;
        int min_rank = -1;
        
        for (int r = 0; r < n_ranks; r++) {
            int dist = all_dists[r * n_global + i];
            if (dist < min_dist) {
                min_dist = dist;
                min_rank = r;
            }
        }
        
        if (min_rank >= 0) {
            recv_parents[i] = all_parents[min_rank * n_global + i];
        }
        
        // Special case: source vertex should have parent -1
        if (i == source - 1) {
            recv_parents[i] = -1;
        }
    }
    
    // Update local vertices with potentially better distances from other processes
    for (int i = 0; i < part->n_local; i++) {
        int global_id = part->local_vertices[i] - 1;
        if (global_id >= 0 && global_id < n_global) {
            int new_dist = recv_dists[global_id];
            
            // Special case: if this is the source vertex, always set distance to 0
            if (part->local_vertices[i] == source) {
                if (part->dist[i] != 0) {
                    part->dist[i] = 0;
                    part->parent[i] = -1;
                    part->affected[i] = true;
                    
                    if (rank == 0 && i < 10) {
                        printf("[DEBUG] Resetting source vertex distance to 0 during sync\n");
                    }
                }
                continue;
            }
            
            // If we find a better distance from another process, update our local vertex
            if (new_dist < part->dist[i]) {
                int old_dist = part->dist[i];
                part->dist[i] = new_dist;
                part->parent[i] = recv_parents[global_id];
                
                // Mark as affected for further propagation
                part->affected[i] = true;
                
                if (rank == 0 && i < 10) {
                    printf("[DEBUG] Local vertex %d updated from sync: %d -> %d\n", 
                           part->local_vertices[i], old_dist, new_dist);
                }
            }
        }
    }
    
    // Now update ghost vertices with the received data
    for (int i = 0; i < part->n_ghost; i++) {
        int global_id = part->ghost_vertices[i] - 1;
        if (global_id >= 0 && global_id < n_global) {
            int new_dist = recv_dists[global_id];
            int new_parent = recv_parents[global_id];
            
            // Special case: if this is the source vertex, always set distance to 0
            if (part->ghost_vertices[i] == source) {
                if (part->ghost_dist[i] != 0) {
                    part->ghost_dist[i] = 0;
                    part->ghost_parent[i] = -1;
                    
                    if (rank == 0) {
                        printf("[DEBUG] Resetting ghost source vertex distance to 0 during sync\n");
                    }
                    
                    // Mark all local vertices connected to this ghost source as affected
                    for (int j = 0; j < part->n_local; j++) {
                        for (int k = 0; k < part->adj_sizes[j]; k++) {
                            if (part->adj[j][2 * k] == source) {
                                part->affected[j] = true;
                                break;
                            }
                        }
                    }
                }
                continue;
            }
            
            if (new_dist != part->ghost_dist[i]) {
                part->ghost_dist[i] = new_dist;
                part->ghost_parent[i] = new_parent;
                
                // Mark all local vertices connected to this ghost vertex as affected
                for (int j = 0; j < part->n_local; j++) {
                    for (int k = 0; k < part->adj_sizes[j]; k++) {
                        if (part->adj[j][2 * k] == part->ghost_vertices[i]) {
                            part->affected[j] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // Clean up
    free(send_dists);
    free(send_parents);
    free(recv_dists);
    free(recv_parents);
    free(all_dists);
    free(all_parents);
}

// Add this function before asynchronous_update to initialize direct distances from source
void initialize_direct_distances() {
    if (rank == 0) {
        printf("[INFO] Initializing direct distances from source vertex %d\n", source);
    }
    
    // Get the process that owns the source vertex
    int source_owner = -1;
    if (source > 0 && source <= n_global) {
        source_owner = part->vertex_owner[source - 1];
    }
    
    // If I'm the owner of the source, initialize distances to its neighbors
    if (rank == source_owner) {
        int source_idx = -1;
        for (int i = 0; i < part->n_local; i++) {
            if (part->local_vertices[i] == source) {
                source_idx = i;
                break;
            }
        }
        
        if (source_idx >= 0) {
            // Set source distance to 0
            part->dist[source_idx] = 0;
            part->parent[source_idx] = -1;
            part->affected[source_idx] = true;
            
            printf("[DEBUG][Rank %d] Setting source vertex %d distance to 0\n", 
                   rank, source);
            
            // Set direct distances to neighbors
            for (int i = 0; i < part->adj_sizes[source_idx]; i++) {
                int neighbor = part->adj[source_idx][2 * i];
                int weight = part->adj[source_idx][2 * i + 1];
                
                // Is this neighbor local or ghost?
                int neighbor_idx = -1;
                if (neighbor - 1 >= 0 && neighbor - 1 < n_global) {
                    neighbor_idx = part->ghost_to_local[neighbor - 1];
                }
                
                if (neighbor_idx >= 0 && neighbor_idx < part->n_local) {
                    // Local neighbor
                    part->dist[neighbor_idx] = weight;
                    part->parent[neighbor_idx] = source;
                    part->affected[neighbor_idx] = true;
                    printf("[DEBUG][Rank %d] Initialized local neighbor %d with distance %d\n", 
                           rank, neighbor, weight);
                } 
                else if (neighbor_idx >= part->n_local && neighbor_idx < part->n_local + part->n_ghost) {
                    // Ghost neighbor
                    int ghost_idx = neighbor_idx - part->n_local;
                    part->ghost_dist[ghost_idx] = weight;
                    part->ghost_parent[ghost_idx] = source;
                    printf("[DEBUG][Rank %d] Initialized ghost neighbor %d with distance %d\n", 
                           rank, neighbor, weight);
                }
            }
        }
    }
    
    // Synchronize all distances
    MPI_Barrier(comm);
    sync_ghost_vertices();
    
    // Now check if any of my local vertices are connected to the source 
    // (in case source is a ghost)
    int source_ghost_idx = -1;
    for (int i = 0; i < part->n_ghost; i++) {
        if (part->ghost_vertices[i] == source) {
            source_ghost_idx = i;
            break;
        }
    }
    
    if (source_ghost_idx >= 0) {
        printf("[DEBUG][Rank %d] Source vertex %d is my ghost vertex with index %d\n", 
               rank, source, source_ghost_idx);
        
        // Set distance of ghost source to 0
        part->ghost_dist[source_ghost_idx] = 0;
        part->ghost_parent[source_ghost_idx] = -1;
        
        // Check all local vertices for connections to source
        for (int i = 0; i < part->n_local; i++) {
            for (int j = 0; j < part->adj_sizes[i]; j++) {
                if (part->adj[i][2 * j] == source) {
                    int weight = part->adj[i][2 * j + 1];
                    if (part->dist[i] > weight) {
                        part->dist[i] = weight;
                        part->parent[i] = source;
                        part->affected[i] = true;
                        printf("[DEBUG][Rank %d] Initialized vertex %d with distance %d (connected to source)\n", 
                               rank, part->local_vertices[i], weight);
                    }
                }
            }
        }
    }
    
    // Count how many vertices have valid distances
    int local_valid = 0;
    for (int i = 0; i < part->n_local; i++) {
        if (part->dist[i] < INF) {
            local_valid++;
        }
    }
    
    int total_valid;
    MPI_Reduce(&local_valid, &total_valid, 1, MPI_INT, MPI_SUM, 0, comm);
    
    if (rank == 0) {
        printf("[INFO] After direct initialization: %d vertices have valid distances\n", 
               total_valid);
    }
    
    MPI_Barrier(comm);
}

// Improved asynchronous update (Algorithm 4)
void asynchronous_update(int async_level) {
    if (rank == 0) {
        printf("[INFO] Starting asynchronous update with level %d\n", async_level);
    }
    
    // Set maximum iterations based on the level
    int max_iterations = 250; // Increase significantly for better propagation
    
    // Initialize all distances from the source
    if (rank == 0) {
        printf("[DEBUG] Reinitializing all distances from source\n");
    }
    
    // First gather the current state: which processes have valid distances
    int local_valid = 0;
    for (int i = 0; i < part->n_local; i++) {
        if (part->dist[i] < INF) {
            local_valid++;
            part->affected[i] = true; // Mark all valid distances as affected
        }
    }
    
    int global_valid = 0;
    MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_SUM, comm);
    
    if (rank == 0) {
        printf("[DEBUG] Current number of vertices with valid distances: %d\n", global_valid);
    }
    
    // Ensure source vertex and its neighbors are properly initialized
    initialize_direct_distances();
    
    // Track and enable propagation
    int iterations = 0;
    bool global_change = true;
    int total_updates = 0;
    
    // Force several iterations to ensure proper propagation
    while (global_change && iterations < max_iterations) {
        iterations++;
        MPI_Barrier(comm);
        bool local_change = false;
        int local_updates = 0;
        
        if (rank == 0 && iterations % 5 == 0) {
            printf("[DEBUG] Iteration %d: Processing affected vertices...\n", iterations);
        }
        
        // Process affected vertices to propagate distances
        #pragma omp parallel for schedule(dynamic) reduction(||:local_change) reduction(+:local_updates)
        for (int i = 0; i < part->n_local; i++) {
            if (part->affected[i]) {
                // Skip infinitely distant vertices - they can't propagate
                if (part->dist[i] == INF) {
                    part->affected[i] = false;
                    continue;
                }
                
                bool vertex_changed = false;
                
                // Special case: source vertex should never change
                if (part->local_vertices[i] == source) {
                    if (part->dist[i] != 0) {
                        part->dist[i] = 0;
                        part->parent[i] = -1;
                        vertex_changed = true;
                        local_change = true;
                        local_updates++;
                    }
                    continue;
                }
                
                // Try to propagate this vertex's distance to all its neighbors
                for (int j = 0; j < part->adj_sizes[i]; j++) {
                    int v_global = part->adj[i][2 * j];
                    int weight = part->adj[i][2 * j + 1];
                    
                    // Skip invalid vertices
                    if (v_global <= 0 || v_global > n_global) continue;
                    
                    // Find local index for neighbor
                    int v_idx = part->ghost_to_local[v_global - 1];
                    
                    // Skip unknown vertices
                    if (v_idx < 0) continue;
                    
                    // Is this a local or ghost vertex?
                    if (v_idx < part->n_local) {
                        // Local vertex - check if we can improve its distance
                        if (part->dist[v_idx] > part->dist[i] + weight) {
                            int old_dist = part->dist[v_idx];
                            part->dist[v_idx] = part->dist[i] + weight;
                            part->parent[v_idx] = part->local_vertices[i];
                            part->affected[v_idx] = true;
                            local_change = true;
                            local_updates++;
                            
                            if (iterations <= 3 && rank == 0) {
                                printf("[DEBUG] Updated vertex %d: %d -> %d (via %d)\n", 
                                      v_global, old_dist, part->dist[v_idx], part->local_vertices[i]);
                            }
                        }
                    } 
                    else if (v_idx < part->n_local + part->n_ghost) {
                        // Ghost vertex - check if we can improve its distance
                        int ghost_idx = v_idx - part->n_local;
                        if (ghost_idx >= 0 && ghost_idx < part->n_ghost) {
                            if (part->ghost_dist[ghost_idx] > part->dist[i] + weight) {
                                int old_dist = part->ghost_dist[ghost_idx];
                                part->ghost_dist[ghost_idx] = part->dist[i] + weight;
                                part->ghost_parent[ghost_idx] = part->local_vertices[i];
                                local_change = true;
                                vertex_changed = true;
                                local_updates++;
                                
                                if (iterations <= 3 && rank == 0) {
                                    printf("[DEBUG] Updated ghost vertex %d: %d -> %d (via %d)\n", 
                                          part->ghost_vertices[ghost_idx], old_dist, 
                                          part->ghost_dist[ghost_idx], part->local_vertices[i]);
                                }
                            }
                        }
                    }
                }
                
                // Only clear the affected flag if no change was made and we're past the first few iterations
                if (!vertex_changed && iterations > 5) {
                    part->affected[i] = false;
                }
            }
        }
        
        // Synchronize ghost vertices between processes
        sync_ghost_vertices();
        
        // Check for new affected vertices from ghost updates
        for (int i = 0; i < part->n_local; i++) {
            for (int j = 0; j < part->adj_sizes[i]; j++) {
                int v_global = part->adj[i][2 * j];
                int weight = part->adj[i][2 * j + 1];
                
                if (v_global <= 0 || v_global > n_global) continue;
                
                int v_idx = part->ghost_to_local[v_global - 1];
                
                // If this is a ghost vertex with finite distance
                if (v_idx >= part->n_local && v_idx < part->n_local + part->n_ghost) {
                    int ghost_idx = v_idx - part->n_local;
                    
                    // If ghost vertex has a valid distance, it might improve this vertex
                    if (ghost_idx >= 0 && ghost_idx < part->n_ghost && 
                        part->ghost_dist[ghost_idx] < INF) {
                        
                        // Check if we can improve this vertex's distance
                        if (part->dist[i] > part->ghost_dist[ghost_idx] + weight) {
                            int old_dist = part->dist[i];
                            part->dist[i] = part->ghost_dist[ghost_idx] + weight;
                            part->parent[i] = v_global;
                            part->affected[i] = true;
                            local_change = true;
                            local_updates++;
                            
                            if (iterations <= 3 && rank == 0) {
                                printf("[DEBUG] Updated local vertex %d from ghost: %d -> %d (via %d)\n", 
                                      part->local_vertices[i], old_dist, part->dist[i], v_global);
                            }
                        }
                        else {
                            // If ghost already has a distance, mark this vertex as affected
                            // to ensure proper propagation
                            part->affected[i] = true;
                        }
                    }
                }
            }
        }
        
        // Check if any process still has changes to propagate
        MPI_Allreduce(&local_change, &global_change, 1, MPI_C_BOOL, MPI_LOR, comm);
        
        // Sum up updates across all processes
        int global_updates = 0;
        MPI_Allreduce(&local_updates, &global_updates, 1, MPI_INT, MPI_SUM, comm);
        total_updates += global_updates;
        
        if (rank == 0 && (iterations <= 10 || iterations % 5 == 0 || global_updates > 0)) {
            printf("[INFO] Async update iteration %d: changes=%d, updates=%d, total=%d\n", 
                  iterations, global_change, global_updates, total_updates);
        }
        
        // Force more iterations at the beginning for better propagation
        if (iterations <= 5) {
            global_change = true;
        }
        
        // If very few updates are happening but we haven't run many iterations yet, continue
        if (iterations < 20 && global_updates < 10 && iterations > 5) {
            global_change = true;
        }
    }
    
    // Do a final sync
    sync_ghost_vertices();
    
    // Count final valid distances
    local_valid = 0;
    for (int i = 0; i < part->n_local; i++) {
        if (part->dist[i] < INF) {
            local_valid++;
        }
    }
    
    global_valid = 0;
    MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_SUM, comm);
    
    if (rank == 0) {
        printf("[INFO] Asynchronous update completed after %d iterations with %d total updates\n", 
              iterations, total_updates);
        printf("[INFO] Final number of vertices with valid distances: %d\n", global_valid);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    double start_time = MPI_Wtime();

    if (rank == 0) {
        printf("[INFO] Starting parallel SSSP with %d processes\n", n_ranks);
    }

    // Read graph and partition
    idx_t* xadj, *adjncy;
    read_graph_and_partition("USA-road-d.NY.gr", &xadj, &adjncy);
    
    if (n_global <= 0 || m_global <= 0) {
        fprintf(stderr, "[ERROR][Rank %d] Invalid graph dimensions: n=%d, m=%d\n", 
                rank, n_global, m_global);
        MPI_Abort(comm, 1);
    }

    // Setup METIS partitioning
    idx_t n_vertices = n_global, n_parts = n_ranks, n_constraints = 1;
    idx_t* part_map = (idx_t*)malloc(n_vertices * sizeof(idx_t));
    if (!part_map) {
        fprintf(stderr, "[ERROR][Rank %d] Failed to allocate part_map\n", rank);
        MPI_Abort(comm, 1);
    }
    
    idx_t edgecut;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
    options[METIS_OPTION_NITER] = 10;
    options[METIS_OPTION_NCUTS] = 1;
    options[METIS_OPTION_SEED] = 42; // Fixed seed for reproducibility
    options[METIS_OPTION_UFACTOR] = 30;
    options[METIS_OPTION_DBGLVL] = 0;

    // Partition the graph
    int ret = METIS_PartGraphKway(&n_vertices, &n_constraints, xadj, adjncy, NULL, NULL, NULL,
                                  &n_parts, NULL, NULL, options, &edgecut, part_map);
    if (ret != METIS_OK) {
        fprintf(stderr, "[ERROR][Rank %d] METIS partitioning failed with code %d\n", rank, ret);
        MPI_Abort(comm, 1);
    }
    if (rank == 0) {
        printf("[DEBUG] METIS partitioning completed, edgecut: %d\n", edgecut);
    }

    // Initialize partition data
    initialize_partition(part_map, xadj, adjncy);
    
    // Read initial SSSP tree
    read_initial_sssp("initial_distances.txt");
    
    // Initialize direct distances from source
    if (rank == 0) {
        printf("[INFO] Initializing direct distances from source after reading initial distances\n");
    }
    initialize_direct_distances();
    
    // Debug source vertex and its neighbors
    if (rank == 0) {
        printf("[INFO] Debugging source vertex %d and its neighbors...\n", source);
    }
    debug_source_vertex();
    
    // Check source vertex location
    int source_owner = -1;
    if (source >= 1 && source <= n_global) {
        source_owner = part->vertex_owner[source - 1];
        if (rank == 0) {
            printf("[INFO] Source vertex %d is owned by rank %d\n", source, source_owner);
        }
    } else {
        if (rank == 0) {
            printf("[WARNING] Invalid source vertex %d (outside range 1-%d)\n", source, n_global);
        }
    }
    
    // Initial synchronization to ensure source is properly initialized
    MPI_Barrier(comm);
    sync_ghost_vertices();
    
    // Initial asynchronous update with higher level to ensure proper initialization
    if (rank == 0) {
        printf("[INFO] Running initial asynchronous update to initialize distances from source\n");
    }
    initialize_direct_distances();
    asynchronous_update(8); // Higher level for initial update

    // Process edge changes
    const char* changes_file = "edge_changes.txt";
    FILE* changes = fopen(changes_file, "r");
    if (!changes) {
        fprintf(stderr, "[ERROR][Rank %d] Cannot open changes file: %s\n", rank, changes_file);
        MPI_Abort(comm, 1);
    }

    char line[MAX_LINE];
    int change_count = 0;
    int successful_changes = 0;
    int skipped_changes = 0;
    
    // Wait for all processes to reach this point
    MPI_Barrier(comm);
    
    if (rank == 0) {
        printf("[INFO] Starting edge change processing...\n");
    }
    
    while (fgets(line, MAX_LINE, changes)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        
        char op;
        int u, v, w;
        int items = sscanf(line, "%c %d %d %d", &op, &u, &v, &w);
        if (items != 4) {
            if (rank == 0) {
                printf("[WARNING] Invalid change format: %s (got %d items)\n", line, items);
            }
            skipped_changes++;
            continue;
        }
        
        if (u < 1 || u > n_global || v < 1 || v > n_global) {
            if (rank == 0) {
                printf("[WARNING] Invalid vertex in change: %s\n", line);
            }
            skipped_changes++;
            continue;
        }
        
        if (op != 'I' && op != 'D') {
            if (rank == 0) {
                printf("[WARNING] Invalid operation in change: %c (must be I or D)\n", op);
            }
            skipped_changes++;
            continue;
        }
        
        // Apply edge change
        apply_edge_change(op, u, v, w);
        
        // Process edge change
        process_edge_changes(op, u, v, w);
        
        change_count++;
        
        // Check for vertices affected in this process
        bool local_affected = false;
        for (int i = 0; i < part->n_local; i++) {
            if (part->affected[i] || part->affected_del[i]) {
                local_affected = true;
                break;
            }
        }
        
        if (local_affected) {
            successful_changes++;
        }
        
        // Print progress
        if (change_count % 10 == 0 && rank == 0) {
            printf("[DEBUG] Processed %d edge changes, %d with local effect, %d skipped\n", 
                   change_count, successful_changes, skipped_changes);
        }
        
        // Synchronize ghost vertices periodically
        if (change_count % 10 == 0) {
            // Add barrier to ensure all processes finish current changes
            MPI_Barrier(comm);
            sync_ghost_vertices();
        }
    }
    fclose(changes);
    
    // Final synchronization of ghost vertices
    MPI_Barrier(comm);
    sync_ghost_vertices();
    
    if (rank == 0) {
        printf("[INFO] Total edge changes processed: %d, successful: %d, skipped: %d\n", 
               change_count, successful_changes, skipped_changes);
    }

    // Run asynchronous update to propagate changes with more iterations
    MPI_Barrier(comm);
    if (rank == 0) {
        printf("[INFO] Running final asynchronous update to propagate all changes\n");
    }
    asynchronous_update(5); // Increase async level to 10 for more thorough propagation

    // Debug final state of source vertex and its neighbors
    if (rank == 0) {
        printf("[INFO] Debugging final state of source vertex and its neighbors...\n");
    }
    debug_source_vertex();

    // Write results to output file
    char output_file[64];
    sprintf(output_file, "parallel_distances_rank%d.txt", rank);
    FILE* outfile = fopen(output_file, "w");
    if (!outfile) {
        fprintf(stderr, "[ERROR][Rank %d] Cannot open output file: %s\n", rank, output_file);
        MPI_Abort(comm, 1);
    }
    
    fprintf(outfile, "# Final distances from source vertex %d (rank %d)\n", source, rank);
    fprintf(outfile, "# Format: <vertex_id> <distance>\n");
    
    int printed = 0;
    for (int i = 0; i < part->n_local; i++) {
        if (part->dist[i] != INF) {
            fprintf(outfile, "%d %d\n", part->local_vertices[i], part->dist[i]);
            printed++;
        }
    }
    
    fprintf(outfile, "# Total local vertices with valid distances: %d\n", printed);
    fclose(outfile);
    
    if (rank == 0) {
        printf("[INFO] Each process has written its distances to parallel_distances_rankN.txt\n");
        printf("[INFO] Use 'cat parallel_distances_rank*.txt > all_distances.txt' to combine\n");
    }
    
    // write a combined output file from rank 0 
    if (rank == 0) {
        // Gather all distances to rank 0
        int* all_dists = (int*)malloc(n_global * sizeof(int));
        int* all_owners = (int*)malloc(n_global * sizeof(int));
        
        // Initialize with INF
        for (int i = 0; i < n_global; i++) {
            all_dists[i] = INF;
            all_owners[i] = -1;
        }
        
        // Set local distances
        for (int i = 0; i < part->n_local; i++) {
            int global_id = part->local_vertices[i] - 1;
            if (global_id >= 0 && global_id < n_global) {
                all_dists[global_id] = part->dist[i];
                all_owners[global_id] = rank;
            }
        }
        
        // Create buffers to receive distances from other ranks
        int* recv_dists = (int*)malloc(n_global * sizeof(int));
        int* recv_owners = (int*)malloc(n_global * sizeof(int));
        
        // Gather distances from all ranks
        for (int r = 1; r < n_ranks; r++) {
            // Initialize receive buffer
            for (int i = 0; i < n_global; i++) {
                recv_dists[i] = INF;
                recv_owners[i] = -1;
            }
            
            // Receive distances from rank r
            MPI_Recv(recv_dists, n_global, MPI_INT, r, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(recv_owners, n_global, MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
            
            // Merge with better distances
            for (int i = 0; i < n_global; i++) {
                if (recv_dists[i] < all_dists[i]) {
                    all_dists[i] = recv_dists[i];
                    all_owners[i] = recv_owners[i];
                }
            }
        }
        
        // Ensure source has distance 0
        if (source - 1 >= 0 && source - 1 < n_global) {
            all_dists[source - 1] = 0;
            
            // Set owner to the process that owns the source
            if (source_owner >= 0) {
                all_owners[source - 1] = source_owner;
            }
        }
        
        // Write combined file
        FILE* combined = fopen("combined_distances.txt", "w");
        if (combined) {
            fprintf(combined, "# Final distances from source vertex %d (combined)\n", source);
            fprintf(combined, "# Format: <vertex_id> <distance> <owner_rank>\n");
            
            int total_printed = 0;
            for (int i = 0; i < n_global; i++) {
                if (all_dists[i] < INF) {
                    fprintf(combined, "%d %d %d\n", i + 1, all_dists[i], all_owners[i]);
                    total_printed++;
                }
            }
            
            fprintf(combined, "# Total vertices with valid distances: %d\n", total_printed);
            fclose(combined);
            printf("[INFO] Wrote combined distances to combined_distances.txt\n");
            
            // Verify some key distances for debugging
            printf("[DEBUG] Checking some distances in the combined output:\n");
            for (int test_vertex : {1, 17203, 27426, 10000, 20000}) {
                if (test_vertex - 1 >= 0 && test_vertex - 1 < n_global) {
                    printf("  Vertex %d: distance = %d, owner = %d\n", 
                           test_vertex, 
                           all_dists[test_vertex - 1], 
                           all_owners[test_vertex - 1]);
                }
            }
        }
        
        free(all_dists);
        free(all_owners);
        free(recv_dists);
        free(recv_owners);
    } else {
        // Other ranks send their distances to rank 0
        int* send_dists = (int*)malloc(n_global * sizeof(int));
        int* send_owners = (int*)malloc(n_global * sizeof(int));
        
        // Initialize with INF
        for (int i = 0; i < n_global; i++) {
            send_dists[i] = INF;
            send_owners[i] = -1;
        }
        
        // Set local distances
        for (int i = 0; i < part->n_local; i++) {
            int global_id = part->local_vertices[i] - 1;
            if (global_id >= 0 && global_id < n_global) {
                send_dists[global_id] = part->dist[i];
                send_owners[global_id] = rank;
            }
        }
        
        // Send to rank 0
        MPI_Send(send_dists, n_global, MPI_INT, 0, 0, comm);
        MPI_Send(send_owners, n_global, MPI_INT, 0, 1, comm);
        
        free(send_dists);
        free(send_owners);
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("[INFO] Total execution time: %.2f seconds\n", end_time - start_time);
    }

    // Clean up
    for (int i = 0; i < part->n_local; i++) {
        free(part->adj[i]);
    }
    free(part->adj);
    free(part->adj_sizes);
    free(part->dist);
    free(part->parent);
    free(part->affected);
    free(part->affected_del);
    free(part->local_vertices);
    free(part->ghost_vertices);
    free(part->ghost_dist);
    free(part->ghost_parent);
    free(part->ghost_to_local);
    free(part->vertex_owner);
    free(part);
    free(xadj);
    free(adjncy);
    free(part_map);

    MPI_Finalize();
    return 0;
}


