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

use crate::format::pb;
use std::fmt;

use crate::encodings::Encoding;

#[derive(Debug)]
pub struct Field {
    pub id: i32,
    pub parent_id: i32,
    pub name: String,
    pub encoding: Option<Encoding>,

    children: Vec<Field>,
}

impl Field {
    pub fn new(pb: &pb::Field) -> Field {
        Field {
            id: pb.id,
            parent_id: pb.parent_id,
            name: pb.name.to_string(),
            encoding: match pb.encoding {
                1 => Some(Encoding::Plain),
                2 => Some(Encoding::VarBinary),
                3 => Some(Encoding::Dictionary),
                _ => None,
            },

            children: vec![],
        }
    }

    fn insert(&mut self, child: Field) {
        self.children.push(child)
    }

    fn field_mut(&mut self, id: i32) -> Option<&mut Field> {
        for field in &mut self.children {
            if field.id == id {
                return Some(field);
            }
            match field.field_mut(id) {
                Some(c) => return Some(c),
                None => {}
            }
        }
        None
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Field({}, id={}, encoding={})",
            self.name,
            self.id,
            match &self.encoding {
                Some(enc) => format!("{}", enc),
                None => String::from("N/A"),
            }
        )?;
        self.children.iter().for_each(|field| {
            write!(f, "{}", field).unwrap();
        });
        Ok(())
    }
}

/// Lance file Schema.
#[derive(Debug)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: &Vec<crate::format::pb::Field>) -> Schema {
        let mut schema = Schema { fields: vec![] };
        fields.iter().for_each(|f| {
            let lance_field = Field::new(f);
            if lance_field.parent_id < 0 {
                schema.fields.push(lance_field);
            } else {
                schema
                    .field_mut(lance_field.parent_id)
                    .map(|f| f.insert(lance_field));
            }
        });
        schema
    }

    fn field_mut(&mut self, id: i32) -> Option<&mut Field> {
        for field in &mut self.fields {
            if field.id == id {
                return Some(field);
            }
            match field.field_mut(id) {
                Some(c) => return Some(c),
                None => {}
            }
        }
        None
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Schema(")?;
        for field in &self.fields {
            write!(f, "{}", field)?
        }
        writeln!(f, ")")
    }
}
