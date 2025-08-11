import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def search_per_group_parallel(queries, embeds, table, max_workers=12):
    """
    Parallel version processing multiple main queries simultaneously
    """
    
    def search_single_group(args):
        """Process one main query group"""
        group_idx, subqueries, subembeds = args
        
        if not subqueries:
            return group_idx, []
        
        per_sub_limit = 20 if len(subqueries) <= 3 else 15
        gathered = []
        
        for subq, subemb in zip(subqueries, subembeds):
            df = (
                table.search(query_type="hybrid")
                    .vector(subemb)
                    .text(subq)
                    .limit(per_sub_limit)
                    .to_pandas()
            )
            texts = df["text"].to_list()
            gathered.extend(texts)
        
        # Deduplicate while preserving order
        unique_contexts = list(dict.fromkeys(gathered))
        return group_idx, unique_contexts
    
    # Prepare arguments with group indices
    args_list = [
        (i, subqueries, subembeds) 
        for i, (subqueries, subembeds) in enumerate(zip(queries, embeds))
    ]
    
    # Execute in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(search_single_group, args): args[0] 
            for args in args_list
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            group_idx, contexts = future.result()
            results[group_idx] = contexts
    
    # Reconstruct results in original order
    all_contexts = [results[i] for i in range(len(queries))]
    return all_contexts

def search_per_group(queries, embeds,table):
        """
        queries: list of lists of subquery strings
        embeds: list of lists of embeddings (same shape as queries)
        returns: list of lists; one list of unique context chunks per main query
        """
        all_contexts = []

        for subqueries, subembeds in zip(queries, embeds):
            if not subqueries:
                all_contexts.append([])
                continue

            per_sub_limit = 5 if len(subqueries) < 3 else 3
            gathered = []

            for subq, subemb in zip(subqueries, subembeds):
                df = (
                    table.search(query_type="hybrid")
                        .vector(subemb)
                        .text(subq)
                        .limit(per_sub_limit)
                        .to_pandas()
                )
                texts = df["text"].to_list()
                gathered.extend(texts)

            # deduplicate while preserving order
            unique_contexts = list(dict.fromkeys(gathered))
            all_contexts.append(unique_contexts)

        return all_contexts