import pyarrow as pa
import pyarrow.compute as pc
from typing import Optional, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class TextUppercaseMapper(BaseOperator):
    """
    An operator that converts a specific text field to uppercase.
    
    Category: Mapper
    Description: Practical example of dual-path processing (Item & Arrow).
    """
    # 1. Identity & Configuration
    op_name: str = "mapper_text_uppercase"
    op_version: str = "1.0.0"
    
    # Custom config: which field to process?
    target_col: str = "text"

    def input_schema(self) -> Optional[OperatorSchema]:
        """Declare that this operator expects an item or table with a text field."""
        # For simplicity, we return a basic item schema
        return OperatorSchema.item(schema={"text": str})

    def output_schema(self) -> Optional[OperatorSchema]:
        """Declare the output format remains the same."""
        return self.input_schema()

    # 2. Row-based Path (Item)
    def process_item(self, ctx: OperatorContext, item: dict) -> dict:
        """
        Logic for single Python dicts. 
        Useful for small streams or complex custom Python logic.
        """
        if self.target_col in item and isinstance(item[self.target_col], str):
            item[self.target_col] = item[self.target_col].upper()
        return item

    # 3. Batch-based Path (Arrow)
    def process_arrow_batch(self, ctx: OperatorContext, batch: pa.RecordBatch) -> pa.RecordBatch:
        """
        Logic for high-performance Arrow batches.
        Uses vectorised C++ kernels via pyarrow.compute.
        """
        # Find the index of the column to process
        try:
            idx = batch.schema.get_field_index(self.target_col)
            if idx == -1:
                return batch # Or raise error depending on your policy
            
            # Extract the column
            col_data = batch.column(idx)
            
            # Apply vectorized uppercase operation (extremely fast)
            upper_col = pc.utf8_upper(col_data)
            
            # Replace the old column with the new one
            new_columns = list(batch.columns)
            new_columns[idx] = upper_col
            
            return pa.RecordBatch.from_arrays(new_columns, schema=batch.schema)
            
        except Exception as e:
            # Proper error handling within the context
            raise OperatorError(
                f"Failed to process arrow batch in {self.op_name}: {str(e)}",
                code="ARROW_PROC_ERROR"
            )

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Setup Context
    ctx = OperatorContext(trace_id="test_123")
    
    # 2. Initialize Operator
    op = TextUppercaseMapper(target_col="content")
    
    # 3. Test Item Path
    sample_item = {"content": "hello world", "id": 1}
    result_item = op.process_item(ctx, sample_item)
    print(f"Item Result: {result_item}")
    
    # 4. Test Arrow Path
    if HAS_ARROW:
        table = pa.Table.from_pydict({"content": ["apple", "banana"], "id": [1, 2]})
        batch = table.to_batches()[0]
        result_batch = op.process_arrow_batch(ctx, batch)
        print(f"Arrow Result: {result_batch.to_pydict()}")
