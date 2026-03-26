#!/bin/bash
set -e

# Configuration
NFS_SHARED_DIR=${NFS_SHARED_DIR:-"/nfs/shared"}
DLC_JOB_ID=${DLC_JOB_ID:-"test-job"}
NODE_TYPE=${NODE_TYPE:-"worker"}  # master or worker
NODE_RANK=${NODE_RANK:-0}
TIMEOUT=${TIMEOUT:-300}  # Wait up to 300 seconds for network interface

# Function to get node name
get_node_name() {
    if [ "$NODE_TYPE" = "master" ]; then
        echo "${DLC_JOB_ID}-master-0"
    else
        echo "${DLC_JOB_ID}-worker-${NODE_RANK}"
    fi
}

NODE_NAME=$(get_node_name)

echo "=========================================="
echo "Node IP Registration Script"
echo "=========================================="
echo "NFS_SHARED_DIR: $NFS_SHARED_DIR"
echo "DLC_JOB_ID: $DLC_JOB_ID"
echo "NODE_TYPE: $NODE_TYPE"
echo "NODE_RANK: $NODE_RANK"
echo "NODE_NAME: $NODE_NAME"
echo "=========================================="

# Create NFS shared directory if it doesn't exist
mkdir -p "$NFS_SHARED_DIR"

# Function to get IP address from network interface
get_ip_address() {
    local interface=$1

    if command -v ip &> /dev/null; then
        # Use ip command (modern Linux)
        ip addr show "$interface" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1
    elif command -v ifconfig &> /dev/null; then
        # Use ifconfig (older systems)
        ifconfig "$interface" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1
    else
        echo "[ERROR] Neither 'ip' nor 'ifconfig' command found"
        return 1
    fi
}

# Try to get IP address from common network interfaces
get_ip_from_interfaces() {
    local interfaces=("eth0" "eth1" "en0" "en1" "bond0" "ib0" "ib1")
    local ip=""

    for iface in "${interfaces[@]}"; do
        if ip link show "$iface" &> /dev/null 2>&1; then
            ip=$(get_ip_address "$iface")
            if [ -n "$ip" ]; then
                echo "[INFO] Found IP $ip on interface $iface"
                echo "$ip"
                return 0
            fi
        fi
    done

    # Fallback: try to get any non-loopback IP
    echo "[INFO] Trying fallback method to get non-loopback IP..."
    if command -v ip &> /dev/null; then
        ip=$(ip addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | head -1)
    elif command -v ifconfig &> /dev/null; then
        ip=$(ifconfig | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | head -1)
    fi

    if [ -n "$ip" ]; then
        echo "[INFO] Found IP $ip using fallback method"
        echo "$ip"
        return 0
    fi

    echo "[ERROR] Could not find a valid IP address"
    return 1
}

# Try to get hostname-based IP
get_hostname_ip() {
    if command -v hostname &> /dev/null; then
        hostname_ip=$(hostname -i 2>/dev/null || echo "")
        if [ -n "$hostname_ip" ] && [ "$hostname_ip" != "127.0.0.1" ] && [ "$hostname_ip" != "127.0.1.1" ]; then
            echo "[INFO] Found IP $hostname_ip from hostname"
            echo "$hostname_ip"
            return 0
        fi
    fi

    return 1
}

# Main logic to get IP address
echo "[INFO] Attempting to get node IP address..."

IP_ADDRESS=""
START_TIME=$(date +%s)

while [ -z "$IP_ADDRESS" ]; do
    # Try interface-based method first
    IP_ADDRESS=$(get_ip_from_interfaces)

    # If that fails, try hostname-based method
    if [ -z "$IP_ADDRESS" ]; then
        IP_ADDRESS=$(get_hostname_ip)
    fi

    # Check timeout
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
        echo "[ERROR] Timeout after $TIMEOUT seconds waiting for network interface"
        exit 1
    fi

    if [ -z "$IP_ADDRESS" ]; then
        echo "[INFO] Network interface not ready yet, waiting 2 seconds..."
        sleep 2
    fi
done

# Validate IP address format
if ! echo "$IP_ADDRESS" | grep -qE '^([0-9]{1,3}\.){3}[0-9]{1,3}$'; then
    echo "[ERROR] Invalid IP address format: $IP_ADDRESS"
    exit 1
fi

echo "[SUCCESS] Node IP address: $IP_ADDRESS"

# Create IP file in NFS shared directory
IP_FILE="$NFS_SHARED_DIR/${NODE_NAME}.ip"

echo "[INFO] Writing IP to file: $IP_FILE"
echo "$IP_ADDRESS" > "$IP_FILE.tmp"
mv "$IP_FILE.tmp" "$IP_FILE"

# Verify the file was written correctly
if [ -f "$IP_FILE" ]; then
    WRITTEN_IP=$(cat "$IP_FILE" | tr -d '[:space:]')
    if [ "$WRITTEN_IP" = "$IP_ADDRESS" ]; then
        echo "[SUCCESS] IP file created successfully: $IP_FILE"
        echo "[SUCCESS] Content: $WRITTEN_IP"

        # Set permissions to allow other nodes to read
        chmod 644 "$IP_FILE"
        echo "[INFO] File permissions set to 644"
    else
        echo "[ERROR] IP file content mismatch"
        exit 1
    fi
else
    echo "[ERROR] Failed to create IP file: $IP_FILE"
    exit 1
fi

# List all IP files in the directory for debugging
echo ""
echo "[INFO] Current IP files in $NFS_SHARED_DIR:"
ls -lh "$NFS_SHARED_DIR"/*.ip 2>/dev/null || echo "  (no .ip files found)"

echo ""
echo "=========================================="
echo "Node IP registration completed!"
echo "=========================================="
