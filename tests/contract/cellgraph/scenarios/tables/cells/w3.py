# @write_table rows lake:db.out append
import pyarrow as pa

rows = pa.table({"id": pa.array([snap], type=pa.int64())})
