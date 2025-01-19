from typing import Dict, Any, Union, List

def convert_chromadb_to_pandas_query(where_clause: Dict[str, Any]) -> str:
    """
    Converts a ChromaDB-style where clause to a pandas query string.
    
    Args:
        where_clause (Dict[str, Any]): ChromaDB-style query dictionary
        
    Returns:
        str: Pandas query string
    """
    def _process_operator(op: str, value: Any) -> str:
        op_map = {
            '$eq': '==',
            '$ne': '!=',
            '$gt': '>',
            '$gte': '>=',
            '$lt': '<',
            '$lte': '<=',
            '$in': 'in',
            '$nin': 'not in'
        }
        return op_map.get(op, op)

    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, list):
            return str([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
        return str(value)

    def _build_query(clause: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        if not isinstance(clause, dict):
            return str(clause)

        conditions = []
        
        for key, value in clause.items():
            if key == '$or':
                or_conditions = []
                for subcondition in value:
                    or_conditions.append(f"({_build_query(subcondition)})")
                conditions.append(" | ".join(or_conditions))
                
            elif key == '$and':
                and_conditions = []
                for subcondition in value:
                    and_conditions.append(f"({_build_query(subcondition)})")
                conditions.append(" & ".join(and_conditions))
                
            elif isinstance(value, dict):
                for op, op_value in value.items():
                    pandas_op = _process_operator(op, op_value)
                    formatted_value = _format_value(op_value)
                    conditions.append(f"{key} {pandas_op} {formatted_value}")
                    
            else:
                formatted_value = _format_value(value)
                conditions.append(f"{key} == {formatted_value}")

        return " & ".join(conditions) if len(conditions) > 1 else conditions[0]

    return _build_query(where_clause)

# Example usage with your existing code:
def apply_chromadb_query_to_pandas(df, where_clause, target_key, valid_values, threshold=90):
    """
    Applies a ChromaDB-style query to a pandas DataFrame after fuzzy matching corrections.
    
    Args:
        df: pandas DataFrame
        where_clause: ChromaDB-style query dictionary
        target_key: Field to apply fuzzy matching
        valid_values: List of valid values for fuzzy matching
        threshold: Fuzzy matching threshold
        
    Returns:
        filtered_df: Filtered pandas DataFrame
        replacements: List of fuzzy matching replacements made
    """
    # First apply the fuzzy matching corrections
    corrected_clause, replacements = update_values_in_clause(
        where_clause, target_key, valid_values, threshold
    )
    
    # Convert the corrected ChromaDB query to pandas query
    pandas_query = convert_chromadb_to_pandas_query(corrected_clause)
    
    # Apply the query to the DataFrame
    filtered_df = df.query(pandas_query)
    
    return filtered_df, replacements
