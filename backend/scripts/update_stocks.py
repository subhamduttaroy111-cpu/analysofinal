import os
import sys

# Add parent directory to path to find config if needed, though we are editing it as text
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = r"../data/raw/intraday"
CONFIG_FILE = r"config.py"

def update_stock_list():
    # 1. Get all symbols from data directory
    try:
        files = os.listdir(DATA_DIR)
        symbols = []
        for f in files:
            if f.endswith("_5m.csv"):
                # Extract symbol name (e.g., "RELIANCE_5m.csv" -> "RELIANCE")
                symbol = f.replace("_5m.csv", "")
                # Add .NS suffix if not present (assuming all are NSE)
                if not symbol.endswith(".NS"):
                    symbol += ".NS"
                symbols.append(symbol)
        
        symbols.sort()
        print(f"Found {len(symbols)} unique stocks.")
        
    except Exception as e:
        print(f"Error reading data directory: {e}")
        return

    # 2. Read config.py
    try:
        with open(CONFIG_FILE, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading config.py: {e}")
        return

    # 3. Replace STOCKS list
    new_lines = []
    in_stocks_list = False
    stocks_written = False

    for line in lines:
        stripped = line.strip()
        
        # Detect start of STOCKS list
        if stripped.startswith("STOCKS = ["):
            in_stocks_list = True
            new_lines.append("STOCKS = [\n")
            # Write all new stocks here
            # Format: 'SYMBOL.NS', 'SYMBOL2.NS', ... (wrapping for readability)
            chunk_size = 5 # 5 stocks per line
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i+chunk_size]
                formatted_chunk = ", ".join([f"'{s}'" for s in chunk])
                if i + chunk_size < len(symbols):
                    formatted_chunk += ","
                new_lines.append(f"    {formatted_chunk}\n")
            new_lines.append("]\n")
            stocks_written = True
            continue
        
        # Detect end of existing STOCKS list (look for closing bracket)
        if in_stocks_list:
            if stripped.endswith("]"):
                in_stocks_list = False
            continue
            
        new_lines.append(line)

    # 4. Write back to config.py
    with open(CONFIG_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Successfully updated config.py with {len(symbols)} stocks.")

if __name__ == "__main__":
    update_stock_list()
