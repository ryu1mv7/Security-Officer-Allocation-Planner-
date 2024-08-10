

class Graph:
    def __init__(self, num) -> None: # V: take a list of names of vertices as numbers
        """
        Function description: 
            this is the constractor of graph which contains both Flow network(FN) and Residual network(RN).
            This function takes number of node as input and make lists that contain node(vertex objects) for both network
        Time complexity: O(m+n) where n is the number of security officers and m is the number of companies
        Aux Space complexity: O(n+m)
        """
        self.adjacency_flow_network = [None] * num
        self.adjacency_residual_network = [None] * num
        for i in range(num):
            residual_node =  Vertex(i)
            self.adjacency_residual_network[i] =residual_node #convert RN to FN
            flow_network_node = Vertex(i)
            self.adjacency_flow_network[i] =flow_network_node
            residual_node.residual_node_to_FN_node = flow_network_node #convert RN to FN


    def reset(self):
        """
        Function description: 
            Reset the vertex info and make sure bfs can run again and again
        Time complexity: O(m+n) where n is the number of security officers and m is the number of companies
        Aux Space complexity: O(1)
        """
        for vertex in self.adjacency_residual_network:
            vertex.discovered = False
            vertex.visited = False
            vertex.previous = None
        for vertex in self.adjacency_flow_network:
            vertex.discovered = False
            vertex.visited = False
            vertex.previous = None

    def add_edges(self, edges_list, FN = True, RN = True): #edge_info:  [(u, v, w)...]
        """ 
        Function description: 
            Add directed edges in the FN and RN from input, set of edges.
        Time complexity: O(m+n)  where n is the number of security officers and m is the number of companies
        Aux Space complexity: O(1)
        """
        if FN:
            for edge in edges_list:

                u = edge[0]
                v = edge[1]
                w = edge[2]
                capacity = edge[3]
                edge_obj = Edge(u,v,w,capacity)
                vertex_obj = self.adjacency_flow_network[u]
                vertex_obj.add_edge(edge_obj)
        if RN: # edges for RN: forward: balance = capacity - flow / backward: flow
            for edge in edges_list:
                # backward : has the opposite direction of arrow but the same weight
                # w is the flow of the backward
                u = edge[1]
                v = edge[0] # opposite directions
                w = edge[2]
                capacity = edge[3]
                edge_obj_backward = Edge(u,v,w, capacity)
                vertex_obj = self.adjacency_residual_network[u]
                vertex_obj.add_edge(edge_obj_backward)
                # forward: capacity is the flow of forward
                u = edge[0]
                v = edge[1]
                capacity = edge[3]
                w = (capacity - edge[2]) # balance
                edge_obj_forward = Edge(u,v,w,capacity)
                vertex_obj = self.adjacency_residual_network[u]
                vertex_obj.add_edge(edge_obj_forward)
                edge_obj_backward.to_forward = edge_obj_forward
                edge_obj_forward.to_backward = edge_obj_backward

    def __str__(self):
        """
        Output the id of all vertices in the graph
        Time complexity: O(m+n) where n is the number of security officers and m is the number of companies
        Aux Space complexity: O(1)
        """
        return_string_fn = "Flow Network " + "\n"
        for vertex in self.adjacency_flow_network:
            return_string_fn +=  str(vertex) + "\n" # call the str method in Vertex class
        return_string_rn = "\n" + "Residual Network " + "\n"
        for vertex in self.adjacency_residual_network:
            return_string_rn +=  str(vertex) + "\n" # call the str method in Vertex class
        return return_string_fn + return_string_rn

    def ford_fulkerson(self, source, end):
        """
        Function description:
            functions keep flowing the flow until reaching the max limit.
            Also, this function calculate max flow
        Approach description:
            By using bfs, every time I find a path from source to sink, I get the min flow of all edges and flow the min to FN.
            (details is in the docsting of update_residual_network())
        Time complexity: O(FE) where F is the F is the flow itself, E is the number of edges
            Time complexity analysis:
                Not sure how many times to iterate so I assume iteration is F times
                Then the while block takes O(m+n+E) according to the substrings below.
                So, I do O(m+n+E)・ O(F) = O(FV+ FE), but v<E, so O(FE).
        Aux Space complexity: O(m+n)
            Input space analysis: input are intergers
            Aux space analysis: functions, argumenting_path() takes O(m+n)
        input:
            source: index of source (super source), integer
            end: index of sink (super sink), integer
        Output:
            max_flow: max flow, integer
        
        """
        max_flow = 0
        while self.argumenting_path(source, end)[0] != []:
            info = self.argumenting_path(source, end) # info will have path and min flow in the path
            augmenting_path = info[0]
            min_flow = info[1]
            max_flow += min_flow
            self.update_residual_network(augmenting_path, min_flow) # update 
        return max_flow

    
    def update_residual_network(self, path, min_flow):
        """
        Function description:
            this function modify the flow of FN and RN according to the min value in one path I found in bfs
        Approach description:
            After I found the min flow in one path from bfs(augurmentingpath), I minus the flow of tha value from all edges of the oath(forward)in RN
            Then, weight of backward edges in RN will be changed accordingly.
            Finally, U add that min flow to FN.
        Time complexity: O(L) where L is the number of edges taken
        Aux Space complexity: O(1)
            """
        # Attribute representation
            # FN weight  forward = flow -> w
            # RN: weight forward => w, backward -> capacity
        for i in range(len(path)-1):
            residual_node = path[i]
            flownetwork_node = residual_node.residual_node_to_FN_node
            for edge in residual_node.edges:
                if self.adjacency_residual_network[edge.v] is path[i+1]: # if next node is the correct one
                    #RN: forward
                    edge.w = edge.w - min_flow 
                    back_ward_edge = edge.to_backward
                    #RN: backward
                    back_ward_edge.capacity = back_ward_edge.capacity + min_flow
            #FN: flow
            for edge in flownetwork_node.edges:
                if self.adjacency_flow_network[edge.v] is path[i+1].residual_node_to_FN_node:
                    edge.w += min_flow
                    
    def argumenting_path(self, start, end):
        """
        Function description:
            This function returns path(source to sink) and min weight of all edges if the path exist using backtracking function.
        Approach description:
           Normal bfs modified a little bit. After I get a path, if I don't have end(sink) in it, that means I couldn't each the sink.
        Time complexity: O(m+n+E) where n is the number of security officers and m is the number of companies, E is the number of edges
            Time complexity analysis: Bfs: V is m+n, E is E and also backtracking takes O(m+n)
        Aux Space complexity: O(m+n)
           """
        self.reset()
        path1 = []
        start_v = self.adjacency_residual_network[start]
        queue = []
        queue.append(start_v)
        start_v.discovered = True
        while len(queue) > 0 : # until I visit all vertices, keep traversing
            u = queue.pop(0) # u : current vertex I am at
            u.visited = True # I have visited u and distance is finalised 
            path1.append(u.id)
            if self.adjacency_residual_network[end].visited == True:
                break
            for edge in u.edges: #load all adjacent(connected) edges of u 
                v = edge.v # one of adjacent vertex of u
                v = self.adjacency_residual_network[v]
                if v.discovered == False and edge.w > 0:# I don't push when the weight of edge is 0(treat the edges whose weight is 0 as no edges in BFS)
                    queue.append(v)
                    v.previous = u
                    v.previous_distance = edge.w # has the weight of incoming edge
                    v.discovered = True
        if end not in path1:
            return [], None
        info = self.backtracking(start, end) # O(m+n)
        path = info[0]
        min_flow = info[1]
        return path, min_flow
    
    def backtracking(self,start, end):
        """
        Function description:
            From start(source) and destination information, return the path and min .
        Time complexity: 
            O(m+n) where n is the number of security officers and m is the number of companies
        Aux Space complexity: 
        where n is the number of security officers and m is the number of companies"""
        min_weight = float("inf")
        path = []
        start = self.adjacency_residual_network[start]
        end = self.adjacency_residual_network[end]
        traverse_tree = end
        path.append(end)
        while traverse_tree is not start: # 
            if traverse_tree.previous_distance < min_weight: # if there is smaller, update
                min_weight = traverse_tree.previous_distance # this is where I saved the weight of each edges
            previous = traverse_tree.previous
            path.append(previous)
            traverse_tree =  previous
        path.reverse() 
        return path, min_weight
    
    def create_node(self, preferences, officers_per_org, min_shifts, max_shifts):
        """
        Function description:
            This function apply all pre-processsing to convert circular with demand to FN.
        Approach description:
            0. Circular with demand
            1. Apply demand constraint (flow conservation)
            2. connect edges between nodes 
                (first_node - security- intermidiate node- day_shift, companies-last node)
                intermidiate node is for the constraint, which security can work only one time a day
                I combined shift and day
            3. connect all -tive demand node with super source, 
                       all +tive demand node with super sink
            Index of all nodes:
                ・security officers: 0 to n-1
                ・intermidiate node: n to 2n-1 (n to n+ num_intermidiate_node -1)
                ...etc. brief is in thye comment
                + extra
                    ・super sink/ source
        Time complexity:O(m+n) where n is the number of security officers and m is the number of companies
            Time complexity analysis: 
                interation happen only m times and n times and otherwise constant times like shift(3) and days(30)
        Aux Space complexity: O(m+n) where n is the number of security officers and m is the number of companies 
            Input space analysis: both input,preferences and officers_per_org,   has shift and (companies or securities)
            Aux space analysis:O(m+n+30days + 3 shifts + extra constant suck as super sink and source) = O(m+n)"""
    #build_node
        # num of security officers: n
        n = len(preferences) #index 0 to n-1
        # num of days: 30
        num_days = 30
        num_intermidiate_node = n * num_days #index range: n to n+ num_intermidiate_node -1

        # num of shifts
        num_shifts = 3
        day_shifts = num_days* num_shifts # index n+ num_intermidiate_node to n+ num_intermidiate_node+90 -1
        # num of companies: m
        m = len(officers_per_org) #index n+ num_intermidiate_node+90 to n+ num_intermidiate_node+90 + m -1
        min = min_shifts
        extra = 10 #index n+ num_intermidiate_node+90 + m to (n+ num_intermidiate_node+90 + m) + 10 -1
        #first_one: index: n+ num_intermidiate_node+90 + m -> node that is connected to all offciers
        #last_node: index n+ num_intermidiate_node+90 + m +1
        #graph = Graph(n+m + day_shifts+ num_intermidiate_node+ extra)
        #--------------------------------------------------------------------------
        #1. Apply demand constraint (flow conservation)
        
        #remove lower bound and make circle with demand(all lower bounds are 0)
        #demand of each of all security officers will be -min
        total_min = 0
        for so_index in range(0, n):
            total_min = total_min - min
            self.adjacency_flow_network[so_index].demand =  - min
        self.adjacency_flow_network[n+m + day_shifts+ num_intermidiate_node].demand =  total_min

        #demand of each of all companies will be the ""(sum of req ppl for each shift ) * num_days(30)""
        i = 0
        total_sink_demand = 0
        for c_index in range(n+ num_intermidiate_node+day_shifts, n+ num_intermidiate_node+day_shifts + m):
            total_req_ppl_all_shifts = officers_per_org[i][0] + officers_per_org[i][1] + officers_per_org[i][2]
            demand_comp = total_req_ppl_all_shifts * num_days
            total_sink_demand += demand_comp
            self.adjacency_flow_network[c_index].demand = demand_comp
            i+= 1
        self.adjacency_flow_network[n+m + day_shifts+ num_intermidiate_node].demand -=  total_sink_demand
        
        #--------------------------------------------------------------------------------------------------------------------------------
        
        #2. connect edges 
        edges = []
        
        #b/w first_one and security officers -> input (n+ num_intermidiate_node+90 + m, 0 to n-1, 0, max_shifts)
        for so_index in range(0, n):
            edges.append((n+ num_intermidiate_node+day_shifts + m, so_index,0,max_shifts))
        #b/w security officers and intermidiate node -> input (0 to n-1, n to n+ num_intermidiate_node-1, 0, 1)
        for so_index in range(0,n):
            for day_index in range(num_days):
                intermidiate_index = num_shifts + so_index * num_days +day_index
                edges.append((so_index, num_shifts + so_index * num_days + day_index, 0,1))
        #b/w intermidiate node and day_shift
        shift_day_index = n+ num_intermidiate_node
        run_time = 0
        inter_index = 0
        for intermidiate_index in range(n, n+ num_intermidiate_node):  
            if run_time == num_days:
                shift_day_index = n+ num_intermidiate_node
                run_time = 0
                inter_index += 1
            for j in range(3):   # 3 shifts
                if preferences[inter_index][j] == 1:
                    edges.append((intermidiate_index,j+shift_day_index, 0, 1))
            shift_day_index+= 3
            run_time+= 1

        #b/w day_shift and companies -> input (n+ num_intermidiate_node to num_intermidiate_node+90, ) 
        for comp_index in range(n+ num_intermidiate_node+day_shifts, n+ num_intermidiate_node+day_shifts+m):
            for day_shift_index in range(n+ num_intermidiate_node,n+ num_intermidiate_node+day_shifts ):
                req_num_ppl = officers_per_org[comp_index-(n+ num_intermidiate_node+day_shifts)][(day_shift_index-(n+ num_intermidiate_node))%3]
                edges.append((day_shift_index,comp_index, 0, req_num_ppl))
                
        #b/w companies and last_node input (company_index, last_node, 0, demand of company)
        for comp_index in range(n+ num_intermidiate_node+day_shifts, n+ num_intermidiate_node+day_shifts+m):
            #capacity = self.adjacency_flow_network[comp_index].demand
            total_req_ppl_all_shifts = officers_per_org[comp_index-(n+ num_intermidiate_node+day_shifts)][0] + officers_per_org[comp_index-(n+ num_intermidiate_node+day_shifts)][1] + officers_per_org[comp_index-(n+ num_intermidiate_node+day_shifts)][2]
            #print(total_req_ppl_all_shifts)
            capacity = total_req_ppl_all_shifts * num_days
            edges.append((comp_index,n+ num_intermidiate_node+day_shifts + m +1, 0, capacity))
            
        self.add_edges(edges) # add all edges (except for source and sink)
        #-----------------------------------------------------------------------------------------------------
        #3. connect all negative demand node with super source, all positive demand node with super sink
        
        edges = []
        super_source_index = n+ num_intermidiate_node+day_shifts + m +2
        super_sink_index = n+ num_intermidiate_node+day_shifts + m +3
        for node in self.adjacency_flow_network:
            if node.demand < 0: # demand is negative -> connect with super source
                node_id = node.id
                edges.append((super_source_index,node_id,0, -node.demand))
                node.demand = 0
            elif node.demand >0:
                edges.append((node.id,super_sink_index, 0, node.demand))
                node.demand = 0
        self.add_edges(edges) # connect super sink and super node
        
        
        return super_source_index, super_sink_index
    
    def bfs(self, start, end):
        """
        Function description: 
        This function returns a list of 4 necessary info for allocation
        [x,y,z,w] x is the index of securities, 
                  y is the intermidiate node
                  z is the day_shift 
                  w is the company index
        Time complexity: O(E) where is the number of edges
        Auxiliary space complexity: O(1)
        """
        path = []
        start_v = self.adjacency_flow_network[start]
        path.append(start_v.id)
        start_v.discovered = True
        node = start_v
        for _ in range(3): # Only need 3 info (index of security, day_shift, companies) for allocation
            for edge in node.edges: 
                v = edge.v 
                v = self.adjacency_flow_network[v]
                if edge.w > 0: # flow exist -> assigned
                    path.append(v.id)
                    v.previous = self.adjacency_flow_network[edge.u]
                    v.visited = True
                    node = v
                    edge.w-= 1 # if I go through the pass, minus one, then next time if weight is 0, I don't go the same pass again
                    break # this allow to go to next node no matter what
        return path

def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
    Function description: 
           given a input, this function returns allocations where allocations[i][j][d][k] is equal to 1 
           i: index of security officer 
           j: index of companies
           d: days
           k: shifts
           Otherwise, return None
        Approach description:
            instanciate a graph class and do pre-process for FN to be runned using fold fulkerson method.
            Then I prepare the list for allocation.
            After I applied to fold fulkerson method, I need to find which edges has weight of 1 for scheduling.
            To find that I used special bfs(not bfs actually), to traverse the edge which weight is 1 in FN.
            I traverse employee -> intermidiate node -> day_shift -> company
            After I get index of each nodes above, I converted them into index which is usable for allocation list.
            The info I got was all for "equal to 1" since I travesed the edges where weight is 1, which should be scheduled.
            I return None when the FN is not feasible. In other words, total number of security offciers  and the total from from source, then if not equal, return None
        Time complexity: O(m*n*n) where n is the number of security officers and m is the number of companies
        Space Complexity:O(m+n) where n is the number of security officers and m is the number of companies
            Input Space complexity analysis: Input takes O(m+n)
            Aux Space complexity analysis: all functions doesn't exceed O(m+n)
        Input: 
            preferences: a list of lists
            officers_per_org: a list of lists
            min_shifts: integer
            max_shifts:integer
        Output:
            allocation: a list that contains allocation of officer
            
            
    """
    #build_node
    # num of security officers: n
    n = len(preferences) #index 0 to n-1
    # num of days: 30
    num_days = 30
    num_intermidiate_node = n * num_days #index n to n+ num_intermidiate_node -1
    # num of shifts
    num_shifts = 3
    day_shifts = num_days* num_shifts # index n+ num_intermidiate_node to n+ num_intermidiate_node+90 -1
    # num of companies: m
    m = len(officers_per_org) #index n+ num_intermidiate_node+90 to n+ num_intermidiate_node+90 + m -1
        
    extra = 10 #index n+ num_intermidiate_node+90 + m to (n+ num_intermidiate_node+90 + m) + 10 -1
    #first_one: index: n+ num_intermidiate_node+90 + m -> node that is connected to all offciers
    #last_node: index n+ num_intermidiate_node+90 + m +1
    #graph = Graph(n+m + day_shifts+ num_intermidiate_node+ extra)
    min = min_shifts
    max = max_shifts
    
    graph = Graph(n+m + day_shifts+ num_intermidiate_node+ extra)
    super_source_n_sink = graph.create_node(preferences,officers_per_org, min, max)
    start = super_source_n_sink[0]
    end = super_source_n_sink[1]
    max_flow = graph.ford_fulkerson(start, end)
    
    #check feasible or not 
    total_num_ppl = 0
    for j in range(m):
        for k in range(num_shifts):
            total_num_ppl += officers_per_org[j][k] * num_days 
    total_flow = 0
    for edge in graph.adjacency_flow_network[start].edges:
        total_flow += edge.w
        
    if total_flow != max_flow:
        return None
    
    
    #-------------------------------------------------------------------------
    # create the list for allocation
    #allocation = [[[[0 for _ in range(num_shifts)] for _ in range(num_days)] for _ in range(m)] for _ in range(n)] #O(mn)
    allocation = [[]]* n
    for i in range(n):
        allocation[i] = [[]] * m
        for j in range(m):
            allocation[i][j] = [[]] * 30
            for k in range(30):
                allocation[i][j][k] = [0, 0, 0] 
    
    
    path_list = [] # this will contain all information of employee that will be assigned to shift
    
    for so_index in range(0,n):
        for _ in range(num_days):
            path = graph.bfs(so_index, end) # using special bfs, I will keep track the path where flow is 1 in FN
            if len(path) == 4 and path[3] > (n+ num_intermidiate_node+day_shifts): #filter abnormal values
                path_list.append(path)# list should contain info for i,j,d,k
    #There is something wrong in the logic of bfs above so I will fail some test cases........
    
    
    for path in path_list:
        # expression to calculate each values from the index
        try:
            i = path[0]
            j = path[3] - (n+ num_intermidiate_node+day_shifts)
            d = (path[2] - ( n+ num_intermidiate_node))// 3 # day
            k =  (path[2] - ( n+ num_intermidiate_node)) % 3  # shift
            allocation[i][j][d][k] = 1
        except:
            return None
        
    
    return allocation

class Vertex:
    def __init__(self, id) -> None:
        """
        Function description: Constructor of Vertex class.
        Input: id of an integer 
		Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        # Basic info of Vertex
        self.id = id # name of vertex(number)
        self.edges = [] # a list of edges connected with the Vertex
        self.discovered = False
        self.visited = False
        #distance: how much the Vertex is far away from source
        self.distance = float("inf") # for dijkstra, initialize by inf
        #for backtracking: where I was from
        self.previous = None
        self.previous_distance = 0 # store the distance(weight) b/w u to v
        self.residual_node_to_FN_node = None
        self.demand = 0 # demand


    def add_edge(self, edge):
        """
        Function description: Add edges to particular vertex
        Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        self.edges.append(edge)


    def __str__(self):
        """
        Function description: output outgoing edges of particular vertex
        Time complexity: O(n) where n is the number of outgoing edges or adjacent vertices
        Auxiliary space complexity: O(1)
        """
        return_string = "Vertex " + str(self.id) 
        for edge in self.edges:
            return_string = return_string +  "\n" +" -----> " + str(edge) 
        return return_string
    

class Edge:
    def __init__(self, u, v, w, capacity) -> None:
        """
        Function description: Constructor of edges
        Edge is defined from two vertices (and weight)
        Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        self.u = u # current vertex
        self.v = v #adjacent vertex
        if w <= capacity:
            self.w = w 
            self.capacity = capacity # This is only for flow network
        self.forward = None
        self.backward = None
        #convert to forward, backward
        self.to_forward = None
        self.to_backward = None
        

    def __str__(self):
        """
        Function description: Output edge info (u,v,w)
        Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        return_string = str(self.v) + ","+ "["+ str(self.w)+"/"+ str(self.capacity) + "]"
        return return_string 
