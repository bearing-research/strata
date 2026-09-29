import java.util.Map;
import java.util.UUID;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.deletes.EqualityDeleteWriter;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.rest.RESTCatalog;

/**
 * Commits one Parquet equality delete file through Iceberg's own Java writer and a
 * RowDelta, as Flink's upsert sink does. Used by tests/test_lake_catalogs_integration.py.
 *
 * <p>Usage: EqualityDeletes <catalog uri> <table> <long column> <value>...
 */
public class EqualityDeletes {
  public static void main(String[] args) throws Exception {
    RESTCatalog catalog = new RESTCatalog();
    catalog.initialize("rest", Map.of("uri", args[0]));
    Table table = catalog.loadTable(TableIdentifier.parse(args[1]));
    Schema keys = table.schema().select(args[2]);
    OutputFile out = table.io().newOutputFile(table.location() + "/data/eq-" + UUID.randomUUID() + ".parquet");
    EqualityDeleteWriter<Record> writer = Parquet.writeDeletes(out)
        .forTable(table)
        .rowSchema(keys)
        .createWriterFunc(GenericParquetWriter::create)
        .set("write.delete.parquet.compression-codec", "uncompressed")
        .overwrite()
        .equalityFieldIds(keys.findField(args[2]).fieldId())
        .buildEqualityWriter();
    try (writer) {
      for (int i = 3; i < args.length; i++) {
        Record row = GenericRecord.create(keys);
        row.setField(args[2], Long.parseLong(args[i]));
        writer.write(row);
      }
    }
    table.newRowDelta().addDeletes(writer.toDeleteFile()).commit();
    System.out.println("committed " + writer.toDeleteFile().path());
  }
}
