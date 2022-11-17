//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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
