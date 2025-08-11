# import redis

# r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# table_name = "National_Parivar"
# question = "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
# key = f"{table_name}:{question.strip()}"

# r.delete(key)
# print(f"Deleted cache for: {key}")


import redis

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Specify the table/document name
table_name = "UNITED_insurance"

# Find all keys starting with this table/document name
pattern = f"{table_name}:*"
keys = r.keys(pattern)

if keys:
    deleted_count = r.delete(*keys)
    print(f"Deleted {deleted_count} cache entries for '{table_name}'")
else:
    print(f"No cache entries found for '{table_name}'")

