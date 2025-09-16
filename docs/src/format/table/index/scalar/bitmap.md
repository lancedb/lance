# Bitmap Index

Bitmap indices use bit arrays to represent the presence or absence of values,
providing extremely fast query performance for low-cardinality columns.

A bitmap index creates a bit vector for each distinct value in a column where:
- Each bit position corresponds to a row
- A bit is set to 1 if the row contains that value
- Boolean operations on bitmaps enable fast query execution

Bitmap indices excel at:
- **Low cardinality columns**: Gender, status, category
- **Set operations**: `WHERE status IN ('active', 'pending')`
- **Boolean queries**: `WHERE is_premium AND region = 'US'`
- **Count queries**: `COUNT(*) WHERE category = 'electronics'`
- **Multi-dimensional filtering**: Combining multiple predicates

## Structure
