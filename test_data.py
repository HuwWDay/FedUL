from fedULapp.task import prepare_data

print("Starting data load test...")
try:
    # We will just try to load the data exactly as a client would
    prepare_data()
    print("✅ Success! The data loaded perfectly.")
except Exception as e:
    print("❌ CRASH DETECTED:")
    import traceback
    traceback.print_exc()