import networkx
from operator import itemgetter
from PyGNA import Utility
from PyGNA import NetworkFrames
import pickle
import copy
import random
import math
import pylab
import sys


class GTrieNode:
    def __init__(self):
        self.depth = 0
        self.is_leaf = False
        self.label = None
        self.graph = networkx.Graph()
        self.parent_sym_conditions = []
        self.final_sym_condition = []
        self.out_links = []
        self.in_links = []
        self.out_link_states = []
        self.in_link_states = []
        self.node_states = []
        self.children = []
        self.parent = None
        self.match_count = 0
        self.probability = 1.
    
    def setDepth(self, depth):
        self.depth = depth
        
    def getDepth(self):
        return self.depth
    
    def setLeaf(self, isleaf):
        self.is_leaf = isleaf
    
    def isLeaf(self):
        return self.is_leaf
    
    def setLabel(self, label):
        self.label = label
        
    def getLabel(self):
        return self.label
    
    def setGraph(self, graph):
        self.graph = graph.copy()
        
    def getGraph(self, copy=False):
        if copy:
            return self.graph.copy()
        else:
            return self.graph
    
    def setOutLinks(self, links):
        self.out_links = links
    
    def getOutLinks(self):
        return self.out_links
    
    def setOutLinkStates(self, links):
        for index in range(0,len(links)):
            if links[index] == 1.0:
                self.out_link_states.append(self.graph.edge[self.graph.nodes()[self.depth]][self.graph.nodes()[index]])
            else:
                self.out_link_states.append({})
                
    def getOutLinkStates(self):
        return self.out_link_states
    
    def setInLinks(self, links):
        self.in_links = links
        
    def getInLinks(self):
        return self.in_links
    
    def setInLinkStates(self, links):
        for index in range(0,len(links)):
            if links[index] == 1.0:
                self.in_link_states.append(self.graph.edge[self.graph.nodes()[index]][self.graph.nodes()[self.depth]])
            else:
                self.in_link_states.append({})   
                
    def getInLinkStates(self):
        return self.in_link_states
    
    def setNodeStates(self, states):
        self.node_states = states
        
    def getNodeStates(self):
        return self.node_states
    
    def setParentSymConditions(self, conditions):
        self.addParentSymConditions(conditions)
        parent = self.getParent()
        if parent != None:
            while parent.getDepth() >= 1:
                new_conditions = []
                for condition in conditions:
                    if condition[0] in parent.getGraph().nodes() and condition[1] in parent.getGraph().nodes():
                        new_conditions.append(condition)
                parent.addParentSymConditions(new_conditions)
                parent = parent.getParent()
        
    def addParentSymConditions(self, conditions):
        self.parent_sym_conditions.append(conditions)
        
    def getParentSymConditions(self):
        return self.parent_sym_conditions
    
    def setFinalSymCondition(self, condition):
        self.final_sym_condition.append(condition)
        
    def getFinalSymCondition(self):
        return self.final_sym_condition
        
    def insertChild(self, child):
        child.depth = self.depth + 1
        self.children.append(child)
        
    def getChildren(self):
        return self.children
    
    def setParent(self, parent):
        self.parent = parent
        
    def getParent(self):
        return self.parent
    
    def getMatchCount(self):
        return self.match_count
    
    def incMatchCount(self):
        self.match_count += 1
        
    def clearMatchCount(self):
        self.match_count = 0
    
    def getProbability(self):
        return self.probability
    
    def setProbability(self, probability):
        self.probability = probability
        
    def areNodeStatesEqual(self, graph, k):
        if self.getGraph().node[self.getGraph().nodes()[k]] == graph.node[graph.nodes()[k]]:
            return True
        else:
            return False
        
    def areEdgeStatesEqual(self, graph, k):

        for first in range(k):
            for second in range(k):
                if self.getGraph().nodes()[first] in self.getGraph().edge and \
                   self.getGraph().nodes()[second] in self.getGraph().edge[self.getGraph().nodes()[first]]:
                    if not (graph.nodes()[first] in graph.edge and 
                            graph.nodes()[second] in graph.edge[graph.nodes()[first]] and
                            self.getGraph().edge[self.getGraph().nodes()[first]][self.getGraph().nodes()[second]] == 
                            graph.edge[graph.nodes()[first]][graph.nodes()[second]]):
                        return False
        
        return True
    
    def areConditionsRespectedWeak(self, vertices):   
        valid_list = []
        if len(vertices) > 0:
            for conditions in self.parent_sym_conditions:
                cond_test = []
                for less, more in conditions:
                    less_index = self.graph.nodes().index(less)
                    more_index = self.graph.nodes().index(more)
                    if less_index <= len(vertices)-1 and more_index <= len(vertices)-1 and \
                    vertices[less_index] > vertices[more_index]:
                        cond_test.append(False)
                        break
                valid_list.append(False) if False in cond_test else valid_list.append(True)
        else:   
            return True
        
        if len(valid_list) > 0:
            valid = False
            for test_valid in valid_list:
                valid = valid or test_valid
            return valid
        else:
            return True
    
    def areConditionsRespectedStrict(self, vertices, candidate):
        test_vertices = copy.deepcopy(vertices)
        test_vertices.append(candidate)
        valid_list = []
        if len(vertices) > 0:
            for conditions in self.final_sym_condition:
                for less, more in conditions:
                    less_index = self.graph.nodes().index(less)
                    more_index = self.graph.nodes().index(more)
                    if less_index <= len(test_vertices)-1 and more_index <= len(test_vertices)-1 and \
                    test_vertices[less_index] > test_vertices[more_index]:
                        return False
        return True

    def getMinLabelForCurrentPos(self, v_used):
        if v_used == []:
            return 0
        
        if len(self.parent_sym_conditions) > 0:
            min_candidates = []
            target_index = len(v_used)
            for condition_set in self.parent_sym_conditions:
                min_candidate = 0
                condition_index = [(self.graph.nodes().index(less), self.graph.nodes().index(more)) for less,more in condition_set]
                for condition in condition_index:
                    if target_index in condition:
                        if target_index == condition[0]:
                            print "Didn't Expect this!!"
                        else:
                            if min_candidate <= v_used[condition[0]]:
                                min_candidate = int(v_used[condition[0]])+1
                min_candidates.append(min_candidate)
    
            if len(min_candidates) > 0:
                return min(min_candidates)
            else:
                return 0
        else:
            return 0
                    
        
########################################################################
class GTrie:
    """
    A python implementation of the modified version of G-Trie data structure described in 
    "G-tries: an efficient data structure for discovering netowrk motifs" by
    Pedro Ribeiro and Fernando Silva"""

    #----------------------------------------------------------------------
    def __init__(self, include_null_graph=True):
        """Constructor"""
        self.root = GTrieNode()
        self.utility = Utility.utility()
        self.null_graph = include_null_graph
        self.matches = []
        self.dict_matches = {}
        self.max_matches = sys.maxint
    
    def getMatches(self, labels=False):
        if labels:
            return self.dict_matches
        else:
            return self.matches
        
    def clearMatchCount(self, node):
        if node.isLeaf():
            node.clearMatchCount()
            
        for child in node.getChildren():
            self.clearMatchCount(child)
            
    def setMaxMatches(self, maximum):
        self.max_matches = maximum
        
    def setProbability(self, node, probability):
        if probability == []:
            node.setProbability(1)
        else:
            node.setProbability(probability[node.getDepth()])
        for child in node.getChildren():
            self.setProbability(child, probability)
    
    def updateProbabilities(self, node):
        parent = node.getParent()
        if parent != None:
            updateParent = True
            
            for child in parent.getChildren():
                if child.getMatchCount() < self.max_matches:
                    updateParent = False
                    break
                
            if updateParent:
                parent.setProbability(0.)
                self.updateProbabilities(parent)
                
    def get_subgraphs(self, node=None, subgraphs=[]):
        if node is None:
            self.get_subgraphs(self.root, subgraphs)
            return subgraphs
        else:
            if node.isLeaf():
                subgraphs.append(node.getGraph(copy=True))
            for child in node.getChildren():
                self.get_subgraphs(child, subgraphs)
                        
    def read(self, path):
        self.root = pickle.load(open(path, "rb"))
        
    def write(self, path):
        pickle.dump(self.root, open(path,"wb"))
        
    def GTCannon(self, Graph):
        """Turn graph into canonical form
        Note: Relies on NetworkX articulation_points which is restricted to 
        undirected graphs"""
        
        # Handle case where graph is empty.
        if len(Graph.nodes()) == 0:
            return Graph
        
        lex_labeling = self.utility.lexicographicallyLargestLabeling(Graph)
        Graph = networkx.relabel_nodes(Graph, lex_labeling,copy=True)
        retGraph = Graph.copy()
        last_degree = Graph.degree()
        original_degree = Graph.degree()
        canon_label = {}
        
        label = len(Graph.nodes())
        while len(Graph.nodes()) > 0:
            
            articulations = list(networkx.articulation_points(Graph.to_undirected())) \
                if networkx.is_directed(Graph) else list(networkx.articulation_points(Graph))
            current_degrees = temp_degrees = Graph.degree()
            
            #Remove articulation points from consideration
            for nodes in articulations:
                if nodes in temp_degrees:
                    temp_degrees.pop(nodes)
            
            #Sort by degree       
            sorted_degrees = sorted(temp_degrees.iteritems(), key=itemgetter(1))
            
            #Capture min degree
            candidates = []
            u_min = sorted_degrees.pop(0)
            candidates.append(u_min)
            
            #Collect candidates with same degree
            while len(sorted_degrees) > 0 and sorted_degrees[0][1] == u_min[1]:
                candidates.append(sorted_degrees.pop(0))

            #Are there ties?
            if len(candidates) > 1:
                first_tie_candidates = []
                sorted_last_degrees = []
                for pair in candidates:
                    sorted_last_degrees.append((pair[0],last_degree[pair[0]]))
                sorted_last_degrees = sorted(sorted_last_degrees, key=itemgetter(1))
                u_min = sorted_last_degrees.pop(0)
                first_tie_candidates.append(u_min)
                
                while len(sorted_last_degrees) > 0 and sorted_last_degrees[0][1] == u_min[1]:
                    first_tie_candidates.append(sorted_last_degrees.pop())
                
                #Still ties?
                if len(first_tie_candidates) > 1:
                    sorted_original_degree = []
                    for pair in first_tie_candidates:
                        sorted_original_degree.append((pair[0],original_degree[pair[0]]))
                    sorted_original_degree = sorted(sorted_original_degree, key=itemgetter(1))
                    u_min = sorted_original_degree.pop(0)
            
            Graph.remove_node(u_min[0])                                                                   
            canon_label[u_min[0]] = label
            label -= 1
            
        retGraph = networkx.relabel_nodes(retGraph, canon_label, copy=True)
        return retGraph
    
    def GTrieInsert(self, graph, label=None, states=False):
        if not self.root.isLeaf() and self.null_graph:
            self.insertRecursive(networkx.Graph(), [], networkx.adjacency_matrix(networkx.Graph()).todense(),
                                 self.root, 0, label, states)
        components = networkx.connected_components(graph.to_undirected()) \
            if networkx.is_directed(graph) else networkx.connected_components(graph)
        component_len = [1 for x in components if len(x) > 1]
        if len(list(components)) > 1 and sum(component_len) > 1:
            print "Illegal Graph Insert: Graph has more than one connnected component."
            return
        cannonGraph = self.GTCannon(graph.copy())
        matrix = networkx.adjacency_matrix(cannonGraph).todense()
        conditions = self.utility.symmetryConditions(cannonGraph)
        self.insertRecursive(cannonGraph, conditions, matrix, self.root, 0, label, states)
        
    def insertRecursive(self, graph, conditions, matrix, node, k, label, states):
        if k == matrix.shape[0]:
            node.is_leaf = True
            node.setFinalSymCondition(conditions)
            #print "Final - " + str(conditions)
            node.setParentSymConditions(conditions)
            #print "Leaf"
            #pos=networkx.fruchterman_reingold_layout(node.getGraph())
            #networkx.draw(node.getGraph(),pos)
            #networkx.draw_networkx_edge_labels(node.getGraph(), pos)
            #pylab.show()             
            if label != None:
                node.setLabel(label)  
        else:
            row = matrix[k,:k+1].tolist().pop(0)
            column = matrix[:k+1,k].ravel().tolist().pop(0)
            for child in node.children:
                if states:
                    if child.out_links == row and child.in_links == column and \
                    child.areNodeStatesEqual(graph, k) and child.areEdgeStatesEqual(graph, k+1):
                        self.insertRecursive(graph, conditions, matrix, child, k+1, label, states)
                        return
                else:
                    if child.out_links == row and child.in_links == column:    
                        self.insertRecursive(graph, conditions, matrix, child, k+1, label, states)
                        return
                    
            new_child = GTrieNode()
            new_child.setDepth(k)
            new_child.setInLinks(column)
            new_child.setOutLinks(row)
            new_child.setGraph(graph.subgraph(graph.nodes()[:k+1]))
            #new_child.setGraph(graph.subgraph(graph.nodes()[:k]))
            new_child.setNodeStates([graph.node[x] for x in new_child.getGraph(copy=True).nodes()])
            new_child.setInLinkStates(column)
            new_child.setOutLinkStates(row)
            node.insertChild(new_child)
            new_child.setParent(node)
            
            #print "Child."
            #pos=networkx.fruchterman_reingold_layout(new_child.getGraph())
            #networkx.draw(new_child.getGraph(),pos)
            #networkx.draw_networkx_edge_labels(new_child.getGraph(), pos)
            #pylab.show()            
            self.insertRecursive(graph, conditions, matrix, new_child, k+1, label, states)
                
    
    def GTrieMatch(self, graph, probability=[], labels=False, states=False):
        self.clearMatch()
        self.setProbability(self.root, probability)
        self.add_null_match(graph, self.root, labels)
        for child in self.root.getChildren():
            nodes_used = []
            if random.random() <= child.getProbability():
                self.match(graph, child, nodes_used, labels, states)
    
    def add_null_match(self, graph, trie_node, labels):
        if trie_node.getMatchCount() < self.max_matches and \
           trie_node.isLeaf() and self.null_graph:
            self.foundMatch(trie_node, networkx.Graph(), [], labels)
    
    def match(self, graph, trie_node, nodes_used, labels, states):
        matched_vertices = self.matchingVertices(graph, trie_node, nodes_used, states)
        
        #Since there is potentially a cap on matches for a specific trie node, 
        #the matched_vertices are now randomized
        random.shuffle(matched_vertices)
        
        for node in matched_vertices:
            if trie_node.getMatchCount() < self.max_matches and \
               trie_node.isLeaf() and trie_node.areConditionsRespectedStrict(nodes_used, node):
                match = copy.deepcopy(nodes_used)
                match.append(node)
                self.foundMatch(trie_node, graph, match, labels)
            
            for child in trie_node.getChildren():
                if random.random() <= child.getProbability():
                    new_used = copy.deepcopy(nodes_used)
                    new_used.append(node)
                    self.match(graph, child, new_used, labels, states)
                
    def matchingVertices(self, graph, trie_node, nodes_used, states):
        candidates = []
        
        if not trie_node.areConditionsRespectedWeak(nodes_used):
            return candidates
        
        min_value = trie_node.getMinLabelForCurrentPos(nodes_used)

        if nodes_used == []:
            candidates = [x for x in graph.nodes() if x >= min_value]
        else:
            cand_graph = graph.to_undirected() if networkx.is_directed(graph) else graph
            connections = [set(cand_graph.neighbors(x)) for x in nodes_used]
            if trie_node.getGraph().degree(trie_node.getGraph().nodes()[len(nodes_used)]) == 0:
                connections.append(set([x for x, y in graph.degree_iter() if y == 0]))
            connections = list(set.union(*connections))
            connections = [x for x in connections if x >= min_value]
            candidates = [x for x in connections if x not in nodes_used]
            
            #Testing the space reduction
            #candidates.sort(key=lambda x: len(graph.neighbors(x)))
            #candidates = [x for x in candidates if len(graph.neighbors(x)) == len(graph.neighbors(candidates[0]))]
            #candidates = [x for x in candidates if x not in nodes_used]
            #candidates = []
            #if len(connections) > 0:
                #candidates = [x for x in graph.neighbors(connections[0]) if x not in nodes_used]
            
        vertices = []
        for node in candidates:
            cand_test = []
            test_nodes = copy.deepcopy(nodes_used)
            test_nodes.append(node)
            if states:
                if graph.node[node] == trie_node.getNodeStates()[len(nodes_used)]:
                    for i in range(0, len(trie_node.getInLinks())):
                        if ((trie_node.getInLinks()[i] == 1 and node in graph.edge[test_nodes[i]] and 
                             trie_node.getInLinkStates()[i] == graph.edge[test_nodes[i]][node]) or 
                            (trie_node.getInLinks()[i] == 0 and node not in graph.edge[test_nodes[i]])) and \
                            ((trie_node.getOutLinks()[i] == 1 and test_nodes[i] in graph.edge[node] and 
                              trie_node.getOutLinkStates()[i] == graph.edge[node][test_nodes[i]]) or
                            (trie_node.getOutLinks()[i] == 0 and test_nodes[i] not in graph.edge[node])):
                            cand_test.append(True)                    
                        else:
                            cand_test.append(False)
                    if False not in cand_test:
                        vertices.append(node)
            else:
                for i in range(0, len(trie_node.getInLinks())):
                    if ((trie_node.getInLinks()[i] == 1 and node in graph.edge[test_nodes[i]]) or 
                        (trie_node.getInLinks()[i] == 0 and node not in graph.edge[test_nodes[i]])) and \
                        ((trie_node.getOutLinks()[i] == 1 and test_nodes[i] in graph.edge[node]) or
                        (trie_node.getOutLinks()[i] == 0 and test_nodes[i] not in graph.edge[node])):
                        cand_test.append(True)                    
                    else:
                        cand_test.append(False)
                if False not in cand_test:
                    vertices.append(node)                
                     
        return vertices
                        
    def foundMatch(self, node, graph, match, labels):
        
        if node.getMatchCount() == self.max_matches:
            node.setProbability(0.)
            self.updateProbabilities(node)
        
        if node.getMatchCount() < self.max_matches:
            node.incMatchCount()
            if labels:
                if node.getLabel() in self.dict_matches:
                    self.dict_matches[node.getLabel()].append(graph.subgraph(match).copy())
                else:
                    self.dict_matches[node.getLabel()] = []
                    self.dict_matches[node.getLabel()].append(graph.subgraph(match).copy())
            else:
                matchGraph = graph.subgraph(match).copy()
                self.matches.append(matchGraph)
                #print str(matchGraph.nodes()) + str(matchGraph.edges())
            
        
    def clearMatch(self):
        self.matches = []
        self.dict_matches = {}
        self.clearMatchCount(self.root)
        
    def createGTrieWithFour(self):
        four_1 = networkx.Graph()
        four_2 = networkx.Graph()
        four_3 = networkx.Graph()
        four_4 = networkx.Graph()
        four_5 = networkx.Graph()
        four_6 = networkx.Graph()
        three_1 = networkx.Graph()
        three_2 = networkx.Graph()
        four_1.add_nodes_from([1,2,3,4])
        four_2.add_nodes_from([1,2,3,4])
        four_3.add_nodes_from([1,2,3,4])
        four_4.add_nodes_from([1,2,3,4])
        four_5.add_nodes_from([1,2,3,4])
        four_6.add_nodes_from([1,2,3,4])
        three_1.add_nodes_from([1,2,3])
        three_2.add_nodes_from([1,2,3])
        four_1.add_edges_from([(1,4),(2,4),(3,4)])
        four_2.add_edges_from([(1,3),(1,4),(2,4)])
        four_3.add_edges_from([(1,3),(1,4),(2,4),(3,4)])
        four_4.add_edges_from([(1,3),(1,4),(2,3),(2,4)])
        four_5.add_edges_from([(1,3),(1,4),(2,3),(2,4),(3,4)])
        four_6.add_edges_from([(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),])
        three_1.add_edges_from([(1,2), (2,3), (1,3)])
        three_2.add_edges_from([(1,2), (2,3)])
        self.GTrieInsert(four_1)
        self.GTrieInsert(four_2)
        self.GTrieInsert(four_3)
        self.GTrieInsert(four_4)
        self.GTrieInsert(four_5)
        self.GTrieInsert(four_6)      
        self.GTrieInsert(three_1)
        self.GTrieInsert(three_2)
        
    def insertEdgeStateTest(self, correct=False):
        four_1 = networkx.Graph()
        four_2 = networkx.Graph()
        four_3 = networkx.Graph()
        four_4 = networkx.Graph()
        three_1 = networkx.Graph()
        
        four_1.add_nodes_from([1,2,3,4])
        four_2.add_nodes_from([1,2,3,4])
        four_3.add_nodes_from([1,2,3,4])
        four_4.add_nodes_from([1,2,3,4])
        three_1.add_nodes_from([1,2,3])

        four_1.add_edge(1,2,state=2)
        four_1.add_edge(2,3,state=1)
        four_1.add_edge(1,3,state=1)
        four_1.add_edge(1,4,state=1)
        
        four_2.add_edge(1,2,state=2)
        four_2.add_edge(2,3,state=1)
        four_2.add_edge(1,3,state=1)
        four_2.add_edge(1,4,state=1)
        four_2.add_edge(3,4,state=2)
        
        four_3.add_edge(1,2,state=1)
        four_3.add_edge(2,3,state=1)
        four_3.add_edge(3,4,state=1)
        four_3.add_edge(1,4,state=1)
        
        four_4.add_edge(1,2,state=1)
        four_4.add_edge(1,3,state=2)
        four_4.add_edge(1,4,state=1)

        three_1.add_edge(1,2, state=2)
        three_1.add_edge(2,3,state=1)
        three_1.add_edge(1,3,state=1)
        
        if correct:
            self.GTrieInsert(three_1,states=True) 
            self.GTrieInsert(four_1,states=True)
            self.GTrieInsert(four_2,states=True)
            self.GTrieInsert(four_3,states=True)
            self.GTrieInsert(four_4,states=True)
        else:
            self.GTrieInsert(four_1,states=True)
            self.GTrieInsert(four_2,states=True)
            self.GTrieInsert(four_3,states=True)
            self.GTrieInsert(four_4,states=True)            
            self.GTrieInsert(three_1,states=True)
            
    def unconnectedNodeTest(self):
        three_1 = networkx.Graph()        
        three_2 = networkx.Graph() 
        three_3 = networkx.Graph()
        four_1 = networkx.Graph()
        
        three_1.add_nodes_from([1,2,3])
        three_2.add_nodes_from([1,2,3])
        three_3.add_nodes_from([1,2,3])
        four_1.add_nodes_from([1,2,3,4])
        
        three_1.add_edges_from([(1,2)])
        three_3.add_edges_from([(1,2),(2,3),(1,3)])
        four_1.add_edges_from([(1,2),(2,3),(1,3)])
        
        self.GTrieInsert(three_1)
        self.GTrieInsert(three_2)
        self.GTrieInsert(three_3)
        self.GTrieInsert(four_1)
        
    def realDataTest(self):
        frames = NetworkFrames.NetworkFrames()
        frames.readGraphML("insert_Graphs.graphML")
        count = 0
        for graph in frames.inputFrames:
            self.GTrieInsert(graph, label=count,states=True)
            count += 1
        
    def insert_from_network_frames(self, path):
        frames = NetworkFrames.NetworkFrames()
        frames.readGraphML(path)
        index = 0
        for frame in frames.getInputNetworks():
            self.GTrieInsert(frame, index)
            index += 1
        
    def empty_graph_test(self):
        graph = networkx.Graph()
        self.GTrieInsert(graph)
        empty_test = networkx.Graph()
        self.GTrieMatch(empty_test)
        num_gtrie_matches = len(self.matches)
        print self.matches          
    
if __name__ == "__main__":

    empty_tree = GTrie()
    empty_tree.empty_graph_test()
    '''directed_trie = GTrie()
    directed_trie.insert_from_network_frames('ff_lhs.graphML')
    directed_network = networkx.readwrite.read_graphml('test_network.graphML')
    pos=networkx.fruchterman_reingold_layout(directed_network)
    networkx.draw(directed_network,pos)
    #networkx.draw_networkx_edge_labels(test_graph, pos)
    pylab.show()
    
    directed_trie.GTrieMatch(directed_network, labels=True)
    #trie.read("GTrieTest.p")
    #import cProfile
    #import StringIO
    #import pstats
    #pr = cProfile.Profile()
    #pr.enable()
    #import time
    #start = time.time()
    #print "Num nodes: " + str(len(test_graph.nodes()))
    #print "Num edges: " + str(len(test_graph.edges()))
    #correct_trie.GTrieMatch(edge_state_test,[1,1,1,1,1], states=True)
    #incorrect_trie.GTrieMatch(edge_state_test,[1,1,1,1,1], states=True)
    #real_data_test.GTrieMatch(test_graph,[1,1,1,.01,.01],labels=True, states=True)
    #elapsed = time.time() - start
    #print "GTrie Elapsed Time: (3,5 complete graph)" + str(elapsed)

    for key in directed_trie.dict_matches.iterkeys():
        print "Length of key: " + str(key) + " is: " + str(len(directed_trie.dict_matches[key]))
        print "Isomorphs: ", [(graph.nodes()) for graph in directed_trie.dict_matches[key]]
    #print len(correct_trie.matches)
    #print len(incorrect_trie.matches)
    #num_gtrie_matches = len(trie.matches)
    #print trie.matches  '''
    

    '''   
    #pr.disable()
    #s = StringIO.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print s.getvalue()        

    grochow_sum = 0
    
    #start = time.time()
    #third_match = trie.utility.findSubgraphInstances(test_graph, four_1)
    #elapsed = time.time() - start
    #print "Elapsed time (Grochow-Kellis four_1): " + str(elapsed)
    #grochow_sum += len(third_match)
    #print num_gtrie_matches - grochow_sum
    
    #start = time.time()
    #third_match = trie.utility.findSubgraphInstances(test_graph, four_2)
    #elapsed = time.time() - start
    #print "Elapsed time (Grochow-Kellis four_2): " + str(elapsed)
    #grochow_sum += len(third_match)
    #print num_gtrie_matches - grochow_sum 

    start = time.time()
    first_match = trie.utility.findSubgraphInstances(test_graph, four_3)
    elapsed = time.time() - start
    print "Elapsed time (Grochow-Kellis four_3): " + str(elapsed)
    grochow_sum += len(first_match)
    print grochow_sum

    #start = time.time()
    #first_match = trie.utility.findSubgraphInstances(test_graph, four_4)
    #elapsed = time.time() - start
    #print "Elapsed time (Grochow-Kellis four_4): " + str(elapsed)
    #grochow_sum += len(first_match)
    #print num_gtrie_matches - grochow_sum   
    
    start = time.time()
    second_match = trie.utility.findSubgraphInstances(test_graph, four_5)
    elapsed = time.time() - start
    print "Elapsed time (Grochow-Kellis four_5): " + str(elapsed)
    grochow_sum += len(second_match)
    print grochow_sum

    
    #start = time.time()
    #first_match = trie.utility.findSubgraphInstances(test_graph, four_6)
    #elapsed = time.time() - start
    #print "Elapsed time (Grochow-Kellis four_6): " + str(elapsed)
    #grochow_sum += len(first_match)
    #print num_gtrie_matches - grochow_sum

    start = time.time()
    first_match = trie.utility.findSubgraphInstances(test_graph, three_1)
    elapsed = time.time() - start
    print "Elapsed time (Grochow-Kellis three_1): " + str(elapsed)
    grochow_sum += len(first_match)
    print grochow_sum   
    
    #start = time.time()
    #second_match = trie.utility.findSubgraphInstances(test_graph, three_2)
    #elapsed = time.time() - start
    #print "Elapsed time (Grochow-Kellis three_2): " + str(elapsed)
    #grochow_sum += len(second_match)
    #print num_gtrie_matches - grochow_sum'''

    print "Done."