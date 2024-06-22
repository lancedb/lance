use bytes::BytesMut;

pub fn create_buffers_from_capacities(capacities: Vec<(u64, bool)>) -> Vec<BytesMut> {
    // At this point we know the size needed for each buffer
    capacities
        .into_iter()
        .map(|(num_bytes, is_needed)| {
            // Only allocate the validity buffer if it is needed, otherwise we
            // create an empty BytesMut (does not require allocation)
            if is_needed {
                BytesMut::with_capacity(num_bytes as usize)
            } else {
                BytesMut::default()
            }
        })
        .collect::<Vec<_>>()
}
