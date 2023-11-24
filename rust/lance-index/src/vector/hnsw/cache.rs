
pub struct CachedStorage<S: HNSWGraphStorage + VectorStorage> {
    storage: S,

    node_cache: RwLock<HashMap<(u8, u64), GraphNode>>,
    node_lru: Mutex<BTreeSet<(u64, u8, u64)>>,

    vector_cache: Cache<u64, Arc<Vec<f32>>>,
}

impl<S: HNSWGraphStorage + VectorStorage> CachedStorage<S> {
    pub fn new(storage: S, cache_size: u64) -> Self {
        Self {
            storage,
            node_cache: RwLock::new(HashMap::new()),
            node_lru: Mutex::new(BTreeSet::new()),
            vector_cache: Cache::new(cache_size),
        }
    }

    fn maybe_evict(&self) {
    }
}

impl<S: HNSWGraphStorage + VectorStorage> HNSWGraphStorage for CachedStorage<S> {
    fn get_entry_points(&self) -> Result<GraphNode> {
        self.storage.get_entry_points()
    }

    fn set_entry_points(&self, node: &GraphNode) -> Result<()> {
        self.storage.set_entry_points(node)
    }

    fn traverse(&self, from: &GraphNode, level: &u8, to: &[u64]) -> Result<Vec<GraphNode>> {
        let mut ret = Vec::with_capacity(to.len());

        for node in to {
            if let Some(node) = self.node_cache.read().unwrap().get(&(*level, *node)) {
                ret.push(node.clone());
                continue;
            }
            let mut stored = self.storage.traverse(from, level, &[*node])?;
            ret.push(stored[0].clone());
            self.node_cache.write().unwrap().insert((*level, *node), stored.pop().unwrap());
        }

        Ok(ret)
    }

    fn put_node(&self, node: &GraphNode) -> Result<()> {
        self.storage.put_node(node)
    }

    fn put_nodes(&self, nodes: &[GraphNode]) -> Result<()> {
        self.storage.put_nodes(nodes)
    }
}

impl<S: HNSWGraphStorage + VectorStorage> VectorStorage for CachedStorage<S> {
    fn get_vectors(&self, id: &[u64]) -> Result<Arc<Vec<f32>>> {
        let mut ret = Vec::with_capacity(id.len());

        for id in id {
            ret.extend(
                self.vector_cache
                    .get_with(*id, || self.storage.get_vectors(&[*id]).unwrap())
                    .as_ref(),
            );
        }

        Ok(Arc::new(ret))
    }

    fn put_vectors(&self, id: &[u64], vector: &Vec<f32>, dims: usize) -> Result<()> {
        self.storage.put_vectors(id, vector, dims)?;
        for (id, vec) in id.iter().zip(vector.chunks(dims)) {
            self.vector_cache.insert(*id, Arc::new(vec.to_vec()));
        }

        Ok(())
    }
}
