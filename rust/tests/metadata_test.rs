use std::fs::Metadata;

#[test]
fn test_locate_batch() {
    use lance::metadata::Metadata;
    let mut metadata = Metadata::new(lance::format::pb::Metadata::default());
    metadata.add_batch_length(10);
    metadata.add_batch_length(20);
    metadata.add_batch_length(30);

    {
        let (batch_idx, idx) = metadata.locate_batch(0).unwrap();
        assert_eq!(batch_idx, 0);
        assert_eq!(idx, 0);
    }

    {
        let (batch_id, idx) = metadata.locate_batch(5).unwrap();
        assert_eq!(batch_id, 0);
        assert_eq!(idx, 5);
    }

    {
        let (batch_id, idx) = metadata.locate_batch(10).unwrap();
        assert_eq!(batch_id, 1);
        assert_eq!(idx, 0);
    }

    {
        let (batch_id, idx) = metadata.locate_batch(29).unwrap();
        assert_eq!(batch_id, 1);
        assert_eq!(idx, 19);
    }

    {
        let (batch_id, idx) = metadata.locate_batch(30).unwrap();
        assert_eq!(batch_id, 2);
        assert_eq!(idx, 0);
    }

    {
        let (batch_id, idx) = metadata.locate_batch(59).unwrap();
        assert_eq!(batch_id, 2);
        assert_eq!(idx, 29);
    }

    {
        assert!(metadata.locate_batch(-1).is_err());
        assert!(metadata.locate_batch(60).is_err());
        assert!(metadata.locate_batch(65).is_err());
    }
}
