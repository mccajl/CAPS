"""
This code is an implementation of the paper 'Clustering via Decision Tree Construction' by Liu et al
The code is adapted from https://github.com/dimitrs/CLTree
"""

from math import sqrt as sqrt
from data import InstanceData
import numpy as np

def _relative_density(dataset):
    return float(dataset.length())/dataset.nr_virtual_points


class CLTree:
    def __init__(self, dataset, min_split=1):

        self.dataset = dataset        
        self.min_split = min_split
        self.clusters = list()
        self.root = None

    def buildTree(self):
        b = BuildTree(self.min_split)
        self.root = b.build(self.dataset)        

    def pruneTree(self, interactive, prune_config):
        if self.root is None:
            self.raiseUndefinedTree()
        
        if interactive:
            desired_height = prune_config['height']
            p = InteractivePrune()
            p.prune(self.root, desired_height, 0)

        else:
            min_y = prune_config['min_y']
            min_rd = prune_config['min_rd']
            min_y = self._transformPercentageToAbsoluteNumber(min_y)        
            p = PruneTree()        
            p.prune(self.root, min_y, min_rd)
                            
    def getClustersList(self, min_nr_instances=1):
        if self.root is None:
            self.raiseUndefinedTree()
        self.clusters = list()
        self._getClustersList(self.root, min_nr_instances) 
        return self.clusters

    def _getClustersList(self, node, min_nr_instances):
        if node.isPrune() and node.getNrInstancesInNode() >= min_nr_instances:            
            self.clusters.append(node)        
            return
        #if node.isLeaf():
        #    self.clusters.append(node)                
        nodes = node.getChildNodes()
        for n in nodes:
            if n: self._getClustersList(n, min_nr_instances)

    def _transformPercentageToAbsoluteNumber(self, min_y):
        return int((float(min_y)/100.0)*self.dataset.length())
        



class BuildTree(object):
    def __init__(self, min_split):       
        self.cutCreator = InfoGainCutFactory(min_split)
        self.datasetSplitter = DatasetSplitter()                        
        self.root = None
            
    def build(self, dataset):
        self._build_tree(dataset, None, 0)        
        return self.root

    def _build_tree(self, dataset, parent, depth):
        bestCut = self._findBestCut(dataset)  
        feat = -1 if bestCut is None else bestCut.idx
    
        dt_node = CLNode(dataset, parent, feat, depth)           
        if parent: parent.addChildNode(dt_node)
        if self._isRootNode(depth): self.root = dt_node
        
        if bestCut is None:
            return
        
        lhs_dataset, rhs_dataset = self._splitDatasetUsingBestCut(dataset, bestCut)
        
        #self._plotCut(bestCut, dataset, lhs_dataset, rhs_dataset)

        if lhs_dataset.length() > 0:
            self._build_tree(lhs_dataset, dt_node, (depth+1))
        if rhs_dataset.length() > 0:
            self._build_tree(rhs_dataset, dt_node, (depth+1))
            
            
    def _isRootNode(self, depth):
        if depth==0 and self.root is None: return True
        
    def _splitDatasetUsingBestCut(self, dataset, bestCut):                           
        dataset.sort(bestCut.idx)        
        idx = dataset.getInstanceIndex(bestCut.inst_id) #This is the idx of the instance. It seems like the tree tries to cut over and over on the 198th index, which is the last entry    
        idx = idx[0]
        lhs_set, rhs_set = self.datasetSplitter.split(dataset, bestCut.idx, bestCut.value, idx+1)                        
        
        for feat in range(dataset.num_feats):        
            if feat == bestCut.idx:
                continue
            minVal = dataset.get_min(feat)
            maxVal = dataset.get_max(feat)
            lhs_set.set_min(feat, minVal)
            lhs_set.set_max(feat, maxVal)
            rhs_set.set_min(feat, minVal)
            rhs_set.set_max(feat, maxVal)
        
        return lhs_set, rhs_set            
            
    def _findBestCut(self, dataset):               
        bestCut = None
        for idx in range(dataset.num_feats):
            dataset.sort(idx)
            di_cut1 = self._calcCut1(dataset, idx)
            if di_cut1 is None: # Ignore dimension
                continue
            
            di_cut2 = self._calcCut2(di_cut1)
            if di_cut2 is None:
                bestCut = self._selectLowerDensityCut(di_cut1, bestCut)
                continue
            
            di_cut3 = self._calcCut3(di_cut1, di_cut2)
            if di_cut3 is None:
                bestCut = self._selectLowerDensityCut(di_cut2, bestCut)
            else:
                bestCut = self._selectLowerDensityCut(di_cut3, bestCut)

        return bestCut
            
    def _calcCut1(self, dataset, idx):
        c = self.cutCreator.cut(dataset, idx) 
        return c

    def _calcCut2(self, di_cut1):   
        lower_density_set = di_cut1.getLowerDensityRegion() 
        c = self.cutCreator.cut(lower_density_set, di_cut1.idx)
        return c                                     
            
    def _calcCut3(self, di_cut1, di_cut2):   
        adjacentRegion = di_cut2.getAdjacentRegion(di_cut1.value, di_cut1.idx)
        otherRegion = di_cut2.getNonAdjacentRegion(di_cut1.value, di_cut1.idx)
                
        di_cut3 = None
        if _relative_density(adjacentRegion) <= _relative_density(otherRegion):
            lower_density_set = di_cut2.getLowerDensityRegion()                                    
            di_cut3 = self.cutCreator.cut(lower_density_set, di_cut2.idx)                                      
        return di_cut3
    
    def _selectLowerDensityCut(self, cut1, cut2):
        if cut1 is None: return cut2
        if cut2 is None: return cut1
        rd1 = cut1.getRelativeDensityOfLowerDensityRegion() 
        rd2 = cut2.getRelativeDensityOfLowerDensityRegion()
        if rd1 < rd2: return cut1
        else: return cut2
    
    """
    def _plotCut(self, bestCut, dataset, lhs_dataset, rhs_dataset):
        if lhs_dataset.length() > 0 or rhs_dataset.length() > 0:
            if myplt.attribute_1 == bestCut.attribute:
                minVal = dataset.get_min(myplt.attribute_2)
                maxVal = dataset.get_max(myplt.attribute_2)
            else:
                minVal = dataset.get_min(myplt.attribute_1)
                maxVal = dataset.get_max(myplt.attribute_1)
            myplt.line(bestCut.attribute, bestCut.value, minVal, maxVal)
            #myplt.draw()
    """


class DatasetSplitter:
    def __init__(self):
        pass
    
    def split(self, dataset, feat_idx, value, idx):
        #print(type(idx))
        l = dataset.instance_values[0:idx]
        r = dataset.instance_values[idx:]
                
        lhs_set = InstanceData(l, dataset.num_feats, dataset.attr_names)
        rhs_set = InstanceData(r, dataset.num_feats, dataset.attr_names)        
                
        rhs_set.set_min(feat_idx, value)
        
        self._splitNrVirtualPoints(dataset, feat_idx, value, lhs_set, rhs_set)
        self._updateVirtualPoints(lhs_set)
        self._updateVirtualPoints(rhs_set)
        
        return lhs_set, rhs_set
        
    def _splitNrVirtualPoints(self, dataset, idx, value, in_set, out_set):
        minV = dataset.get_min(idx)
        maxV = dataset.get_max(idx)
        in_set.nr_virtual_points = int(abs(dataset.nr_virtual_points*((value-minV)/(maxV-minV))))
        out_set.nr_virtual_points = dataset.nr_virtual_points - in_set.nr_virtual_points
        if out_set.nr_virtual_points < 0:
            self.raiseUndefinedNumberOfPoints()
    
    def _updateVirtualPoints(self, data_set):            
        nr_points_in_set = data_set.length()
        data_set.nr_virtual_points = self._calcNumberOfPointsToAdd(nr_points_in_set, data_set.nr_virtual_points)
        data_set.nr_total_instances = nr_points_in_set + data_set.nr_virtual_points
            
    def _calcNumberOfPointsToAdd(self, nr_points_in_node, nr_points_inherited):    
        if nr_points_inherited < nr_points_in_node:
            nr_points = nr_points_in_node
        else:
            nr_points = nr_points_inherited
        return nr_points
    
    def raiseUndefinedNumberOfPoints(self):
        raise DatasetSplitter.UndefinedNumberOfPoints()
    class UndefinedNumberOfPoints(Exception):
        pass    
    
       
class InfoGainCutFactory:
    def __init__(self, min_split):
        self.min_split = min_split
        self.datasetSplitter = DatasetSplitter()

    def cut(self, dataset, idx):
        di_cut = None
        max_info_gain = -1          
        instances = dataset.getInstances(idx)
        #print(instances, idx)
        #print(dataset.instance_values)
        for i, value in enumerate(instances):
            if self._hasRectangle(dataset, idx, value):
                lhs_set, rhs_set = self.datasetSplitter.split(dataset, idx, value, i+1)                
                ig, lset, rset = self._info_gain(dataset, lhs_set, rhs_set)
                if ig > max_info_gain:
                    #print(ig, i) #Successfully finds cuts all over the dataset
                    max_info_gain = ig
                    di_cut = Cut(idx, value, dataset.getId(i), lset, rset)  
        return di_cut
    
    def _hasRectangle(self, dataset, idx, value):
        
        if dataset.get_max(idx) == dataset.get_min(idx): 
            return False
        else:
            if dataset.get_max(idx) == value:
                return False
            else:
                return True

    def _info_gain(self, dataset, lhs_set, rhs_set):                   
        if (lhs_set.nr_total_instances < self.min_split) or (rhs_set.nr_total_instances < self.min_split):
            return -1, lhs_set, rhs_set
    
        ratio_instances_lhs = (float(lhs_set.nr_total_instances)/dataset.nr_total_instances)
        ratio_instances_rhs = (float(rhs_set.nr_total_instances)/dataset.nr_total_instances)
        entropy2 = ratio_instances_lhs*self._calc_entropy(lhs_set) + ratio_instances_rhs*self._calc_entropy(rhs_set)
    
        entropy1 = self._calc_entropy(dataset)
        
        return (entropy1 - entropy2), lhs_set, rhs_set

    def _calc_entropy(self, dataset):
        nr_existing_instances = dataset.length()
        total = nr_existing_instances + dataset.nr_virtual_points
        terms = list()
        terms.append((float(nr_existing_instances)/float(total))*sqrt(float(nr_existing_instances)/float(total)))    
        terms.append((float(dataset.nr_virtual_points)/float(total))*sqrt(float(dataset.nr_virtual_points)/float(total)))                
        return sum(terms)*-1
        
class Cut:
    def __init__(self, idx, value, inst_id, lhsset, rhsset):
        self.idx = idx
        self.value = value
        self.inst_id = inst_id
        self.lhs_set = lhsset
        self.rhs_set = rhsset

    
    def __str__(self):
        s = 'Cut: ' + self.idx + "\n"
        s += str(self.lhs_set.attr_names) + "\n"  
        s += " Max lhs:" + str(self.lhs_set.max_values)+ "\n"  
        s += " Min lhs:" + str(self.lhs_set.min_values)+ "\n"
        s += " Max rhs:" + str(self.rhs_set.max_values)+ "\n" 
        s += " Min rhs:" + str(self.rhs_set.min_values)        
        s += '\n--------\n'
        return s
    
                
    def getNonAdjacentRegion(self, value, idx):    
        dataset = self.getAdjacentRegion(value, idx)
        if dataset is self.lhs_set:
            return self.rhs_set
        if dataset is self.rhs_set:
            return self.lhs_set
        return None
        
    def getAdjacentRegion(self, value, idx):
        def getMinimumDistanceFromValue(dataset, idx, value):
            if dataset.length() < 1:
                return float('inf')
            distance1 = abs(dataset.get_max(idx) - value)
            distance2 = abs(dataset.get_min(idx) - value)
            return min(distance1, distance2)
        rhs_distance = getMinimumDistanceFromValue(self.rhs_set, idx, value)
        lhs_distance = getMinimumDistanceFromValue(self.lhs_set, idx, value)

        if lhs_distance < rhs_distance: return self.rhs_set
        else: return self.lhs_set
    
    def getRelativeDensityOfLowerDensityRegion(self):    
        lower_density_set = self.getLowerDensityRegion()                                                    
        r_density = _relative_density(lower_density_set)                
        return r_density

    def getLowerDensityRegion(self):
        if self.lhs_set is None or self.rhs_set is None:
            self.raiseNoRegionsDefined()
            
        if _relative_density(self.lhs_set) > _relative_density(self.rhs_set):
            return self.rhs_set
        else:
            return self.lhs_set  
    
    def raiseNoRegionsDefined(self):
        raise Cut.NoRegionsDefined("hi")
    class NoRegionsDefined(Exception):
        pass    
       
         
class CLNode(object):
    def __init__(self, dataset, parent, feat, depth):
        self.dataset = dataset
        self.parent = parent
        self.feat = feat
        self.children = list()
        self.can_prune = False

    def setPruneState(self, prune):
        self.can_prune = prune

    def isPrune(self):
        return self.can_prune

    def getRelativeDensity(self):
        return _relative_density(self.dataset)*100.0

    def getNrInstancesInNode(self):
        return self.dataset.length()
    
    def addChildNode(self, node):
        self.children.append(node)

    def getChildNodes(self):
        return self.children
    
    def isLeaf(self):
        if len(self.children) == 0: 
            return True
        else: 
            return False
    
    def getInstanceIds(self): #Get all ids of all the data in each cluster
        ids = self.dataset.getInstances(-1)
        ids = np.array(ids, dtype=np.int32)
        return ids
    
    def get_bounds(self, idx):
        return self.dataset.get_max(idx), self.dataset.get_min(idx)

    
    """
    def _getMajorityClassName(self):
        counts = [0] * len(self.dataset.class_names)
        class_idx = dict()
        for i, cls in enumerate(self.dataset.class_names):
            class_idx[cls] = i
        new_dict = dict (zip(self.dataset.class_map.values(),self.dataset.class_map.keys()))            
        for cls in list(self.dataset.getClasses()):
            v = new_dict[cls]
            counts[class_idx[v]] += 1
            
        max_count = -2
        self.max_class = -1 
        for i, c in enumerate(counts):
            if c > max_count:
                max_count = c
                self.max_class = self.dataset.class_names[i]        

        self.percent = int((max_count / float(self.dataset.length()) )* 100)               
        self.misclassified = self.dataset.length() - max_count

    
    def __str__(self):
        attr = list()
        p = self
        while p:
            attr.append(p.feat)
            p = p.parent
        
        self._getMajorityClassName()
        s = 'Node: ' + '\n'
        s += str(self.dataset.length()) + ' instances, ' + str(self.misclassified) + ' misclassified, ' + str(self.percent)+ '% '+ self.max_class
        s += ", " + str(int(self.getRelativeDensity())) + " relative density " + '\n'
        s += "Cuts " + str(set(attr))+ '\n'
        
        self.dataset.calculate_limits()
        for i, name in enumerate(self.dataset.attr_names):
            s += name + ' max: ' + str(self.dataset.get_max(i))+\
                ' min: ' + str(self.dataset.get_min(i))+'\n'
        
        return s
    """

class InteractivePrune(object):
    """
    I want functions that do the following:
    Take desired height as argument, traverse depth first down the tree and
    prune each node at that height.
    When we setPruneState as True, we indicate that the two subtrees below the node can be pruned.
    If we want a tree of height x, with root being height 0, we should set prune to true at depth x-1
    """
    def prune(self, node, height, cur_height=0):
        
        if node.isLeaf():
            node.setPruneState(True)
            return
        
        if cur_height == height - 1:
            node.setPruneState(True)
            return
        else:
            childNodes = node.getChildNodes()
            for n in childNodes:
                self.prune(n, height, cur_height + 1)
            node.setPruneState(False)
            return
        
        



class PruneTree(object):
    def prune(self, node, min_y, min_rd):
        if node.isLeaf():
            #Stop prune at leaf
            node.setPruneState(True)
            return

        childNodes = node.getChildNodes()
        for n in childNodes:
            #Stop prune here if number of instances is too small. Else check children
            if n.getNrInstancesInNode() < min_y:
                node.setPruneState(True)
            else:
                self.prune(n, min_y, min_rd)
                
        if len(childNodes) < 2: #Stop prune if there was not a split
            node.setPruneState(True)
            return
        
        node.setPruneState(False)
        #stop prune if children relative density is large or if children have no instances
        if childNodes[0].isPrune() and childNodes[1].isPrune():
            if childNodes[0].getRelativeDensity() > min_rd and childNodes[1].getRelativeDensity() > min_rd:
                node.setPruneState(True)
            elif childNodes[0].getNrInstancesInNode() == 0:
                node.setPruneState(True)
            elif childNodes[1].getNrInstancesInNode() == 0:
                node.setPruneState(True)
    
