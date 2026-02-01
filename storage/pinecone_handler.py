
from datetime import datetime, timezone
from pinecone import Pinecone
from config.logger import logger

class PineconeHandler:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = "default"
        logger.info(f"Pinecone handler initialized for index: {index_name}")

    def upsert_records(self, records: list[dict], namespace: str = None):
        """
        Upserts records to Pinecone with automatic embedding from text field.
        Records must have '_id' and 'text'. New fields: 'absoluteStartTime', 'absoluteEndTime', 'speaker', 'keywords'.
        """
        namespace = namespace or self.namespace
        try:
            if not records:
                logger.warning("No records provided for upserting")
                return

            processed_records = []
            for record in records:
                if '_id' not in record or 'text' not in record:
                    raise ValueError("Each record must have '_id' and 'text' fields")
                
                # Ensure numeric fields are Python native types
                # Pinecone expects metadata values to be JSON-serializable.
                # Timestamps should be integers (Unix timestamp).
                if 'absoluteStartTime' in record:
                    record['absoluteStartTime'] = int(record['absoluteStartTime'])
                if 'absoluteEndTime' in record:
                    record['absoluteEndTime'] = int(record['absoluteEndTime'])
                if 'start' in record: # Original chunk-relative start/end
                    record['start'] = float(record['start'])
                if 'end' in record: # Original chunk-relative start/end
                    record['end'] = float(record['end'])
                
                # Ensure 'keywords' is a list of strings if present
                if 'keywords' in record and not isinstance(record['keywords'], list):
                    record['keywords'] = [str(record['keywords'])]
                elif 'keywords' not in record:
                    record['keywords'] = [] # Ensure it always exists as a list
                    
                 # Add `createdAt` in ISO8601 format if missing
                if 'createdAt' not in record:
                    record['createdAt'] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


                processed_records.append(record)

            # Batch upsert (max 96 records/batch for text based on typical text embedding limits)
            batch_size = 96
            for i in range(0, len(processed_records), batch_size):
                batch = processed_records[i:i + batch_size]
                self.index.upsert_records(namespace=namespace, records=batch)
                logger.debug(f"Upserted batch of {len(batch)} records to namespace '{namespace}'")

            logger.info(f"Successfully upserted {len(processed_records)} records to namespace '{namespace}'")

        except Exception as e:
            session_id = records[0].get('sessionId', 'Unknown') if records else "Unknown"
            logger.error(f"Pinecone upsert failed for session {session_id}: {e}", exc_info=True)
            raise

    def semantic_search(self, query: str, user_id: str, top_k: int = 10,
                    start_timestamp: int = None, end_timestamp: int = None,
                    speaker: str = None, keywords: list = None) -> list:
        """
        Performs semantic search using Pinecone with a raw text query,
        and optionally applies OR-based sub-filters for time and speaker.
        """
        try:
            # Always filter at least by userId
            search_filter = {"userId": user_id}
            or_clauses = []

            # Build date filter (createdAt ISO format)
            if start_timestamp and end_timestamp:
                start_iso = start_timestamp
                end_iso = end_timestamp
                logger.info(f"Applying createdAt filter candidate: {start_iso} to {end_iso}")
                search_filter.update({"createdAt": {"$gte": start_iso, "$lt": end_iso}})
                
            # Build speaker filter
            if speaker:
                logger.info(f"Applying speaker filter candidate: {speaker}")
                or_clauses.append({"speaker": {"$eq": speaker}})
                    
            # Keywords filter (each keyword becomes its own OR clause)
            if keywords:
                for kw in keywords:
                    or_clauses.append({"keywords": {"$in": [kw]}})
                logger.info(f"Applying keyword OR filter: {keywords}")


            # Attach OR sub-filter if more than one clause
            if len(or_clauses) > 1:
                search_filter["$or"] = or_clauses
            elif len(or_clauses) == 1:
                # Only one filter — merge it directly
                search_filter.update(or_clauses[0])

            # # Keyword filter (AND with others, but you can OR them too if needed)
            # if keywords:
            #     search_filter["keywords"] = {"$in": keywords}
            #     logger.info(f"Applying keyword filter: {keywords}")
            logger.info(f"Final Pinecone search filter: {search_filter}")

            # Build search body
            search_query_body = {
                "inputs": {"text": query},
                "top_k": top_k,
                "filter": search_filter,
            }

            results = self.index.search(
                namespace=self.namespace,
                query=search_query_body,
            )

            logger.info(f"Pinecone search results (raw): {results}")

            # Convert to dict
            if hasattr(results, "to_dict"):
                results = results.to_dict()
            elif hasattr(results, "dict"):
                results = results.dict()

            # Extract hits
            raw_hits = []
            if isinstance(results, dict):
                if "result" in results and "hits" in results["result"]:
                    raw_hits = results["result"]["hits"]
                elif "hits" in results:
                    raw_hits = results["hits"]

            logger.info(f"Extracted raw hits count: {len(raw_hits)}")

            # Normalize
            normalized_hits = []
            for h in raw_hits:
                fields = h.get("fields", {}) if isinstance(h, dict) else {}
                norm = {
                    "_id": h.get("_id"),
                    "_score": h.get("_score"),
                    "sessionId": fields.get("sessionId") or fields.get("session_id"),
                    "chunkNum": fields.get("chunkNum"),
                    "speaker": fields.get("speaker"),
                    "text": fields.get("text"),
                    "start": fields.get("start"),
                    "end": fields.get("end"),
                    "absoluteStartTime": int(fields.get("absoluteStartTime")) if fields.get("absoluteStartTime") else None,
                    "absoluteEndTime": int(fields.get("absoluteEndTime")) if fields.get("absoluteEndTime") else None,
                    "keywords": fields.get("keywords", []),
                    "userId": fields.get("userId"),
                    "createdAt": fields.get("createdAt"),
                }
                normalized_hits.append(norm)

            logger.info(f"Returning normalized hits (count={len(normalized_hits)})")
            
            if normalized_hits:
                logger.info("Skipping Python OR fallback since Pinecone returned hits")
                return normalized_hits

            # --- Fallback post-filter (if Pinecone doesn't honor $or) ---
            if "$or" in search_filter:
                logger.info("Applying fallback OR post-filter in Python")
                final_hits = []
                for nh in normalized_hits:
                    keep = False
                    for clause in search_filter["$or"]:
                        if "createdAt" in clause and nh.get("createdAt"):
                            created_at = nh["createdAt"]
                            if clause["createdAt"]["$gte"] <= created_at < clause["createdAt"]["$lt"]:
                                keep = True
                        if "speaker" in clause and nh.get("speaker") == clause["speaker"]["$eq"]:
                            keep = True
                    if keep:
                        final_hits.append(nh)
                return final_hits

            return normalized_hits

        except Exception as e:
            logger.error(f"Pinecone query failed for user {user_id}: {e}", exc_info=True)
            return []




