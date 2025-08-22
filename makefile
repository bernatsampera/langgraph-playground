.PHONY: dev env

 # Run on dev
dev:
	@echo "Starting backend server..."
	@echo "Backend: http://localhost:2024"
	uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking        

