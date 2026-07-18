import os
import sys
import json
import socket
import struct
import hmac
import hashlib
import argparse
import threading
from datetime import datetime
from typing import Dict, Any

# Environment Routing - Overridden at runtime by the cluster orchestration layer
DB_CONNECTION_STRING = os.environ.get("DATABASE_URL")
SHARED_SECRET = os.environ.get("ENV_HMAC_SIGNING_KEY")
TARGET_PORT = int(os.environ.get("INFRA_STREAM_PORT", "7790"))

class QuantumDistributedStreamer:
    def __init__(self):
        if not SHARED_SECRET:
            raise ValueError("CRITICAL_CONFIGURATION_ERROR: ENV_HMAC_SIGNING_KEY must be set.")
        self.secret_bytes = SHARED_SECRET.encode('utf-8')

    def _anchor_receipt_to_cluster(self, peer: str, payload_hash: str) -> None:
        """
        Pushes a structural log trace to stdout as an immutable JSON string.
        Log scrapers ingest this block directly into the core production data lake.
        """
        audit_payload = {
            "event": "QUANTUM_RECEIPT_ANCHOR",
            "timestamp": datetime.utcnow().isoformat(),
            "peer": peer,
            "payload_hash": payload_hash,
            "cra_manifest_status": "CRA_PROTOCOL_v2.1_DISTRIBUTED_ANCHOR",
            "storage_target_string": DB_CONNECTION_STRING
        }
        sys.stdout.write(json.dumps(audit_payload) + "\n")
        sys.stdout.flush()

    def _send_framed_msg(self, sock: socket.socket, msg_bytes: bytes) -> None:
        """Prefixes payload with a 4-byte network-order (Big Endian) integer."""
        msg_len = struct.pack('!I', len(msg_bytes))
        sock.sendall(msg_len + msg_bytes)

    def _recv_exact_bytes(self, sock: socket.socket, n: int) -> bytes:
        """Ensures complete stream ingestion across variable network fabrics."""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _recv_framed_msg(self, sock: socket.socket) -> bytes:
        """Extracts length configuration and retrieves the exact subsequent message window."""
        raw_msglen = self._recv_exact_bytes(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        return self._recv_exact_bytes(sock, msglen)

    def stream_package_to_remote(self, host: str, port: int, headers: Dict[str, str], payload: str, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Packages and pushes signed payloads through raw TCP streams across distributed grids.
        """
        envelope = json.dumps({"headers": headers, "payload": payload}).encode('utf-8')
        payload_hash = hashlib.sha256(envelope).hexdigest()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
            self._send_framed_msg(sock, envelope)
            
            ack = self._recv_framed_msg(sock)
            if not ack:
                raise ConnectionResetError("Remote endpoint closed stream without ACK.")
                
            ack_data = json.loads(ack.decode('utf-8'))
            return {
                "status": "SUCCESS",
                "bytes_transmitted": len(envelope),
                "payload_hash": payload_hash,
                "remote_ack": ack_data
            }

    def _process_stream_client(self, client_sock: socket.socket, addr: tuple, verify_sig: bool, anchor_enabled: bool) -> None:
        """Thread-isolated stream boundary evaluator."""
        peer_identity = f"{addr[0]}:{addr[1]}"
        try:
            msg_bytes = self._recv_framed_msg(client_sock)
            if not msg_bytes:
                return
                
            envelope = json.loads(msg_bytes.decode('utf-8'))
            headers = envelope.get("headers", {})
            payload_str = envelope.get("payload", "")
            payload_hash = hashlib.sha256(msg_bytes).hexdigest()
            
            if verify_sig:
                inbound_sig = headers.get("X-Payload-Signature-SHA256")
                computed_sig = hmac.new(self.secret_bytes, msg=payload_str.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
                if not inbound_sig or not hmac.compare_digest(inbound_sig, computed_sig):
                    reject_envelope = json.dumps({"status": "REJECTED", "reason": "CRITICAL_SIGNATURE_MISMATCH"}).encode('utf-8')
                    self._send_framed_msg(client_sock, reject_envelope)
                    return
            
            if anchor_enabled:
                self._anchor_receipt_to_cluster(peer_identity, payload_hash)
                
            ack_envelope = json.dumps({"status": "ACK", "payload_hash": payload_hash}).encode('utf-8')
            self._send_framed_msg(client_sock, ack_envelope)

        except Exception as e:
            error_payload = {"status": "STREAM_THREAD_FAULT", "peer": peer_identity, "error": str(e)}
            sys.stderr.write(json.dumps(error_payload) + "\n")
            sys.stderr.flush()
        finally:
            client_sock.close()

    def launch_ingress_node(self, interface: str = "0.0.0.0", port: int = 7790, verify_sig: bool = False, anchor_enabled: bool = False) -> None:
        """Binds the execution runtime directly to network interfaces for horizontal cloud exposure."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((interface, port))
        server.listen(128)
        
        init_log = {
            "status": "NODE_ONLINE",
            "interface": interface,
            "port": port,
            "security_verification_enforced": verify_sig,
            "cluster_anchoring_active": anchor_enabled
        }
        sys.stdout.write(json.dumps(init_log) + "\n")
        sys.stdout.flush()
        
        try:
            while True:
                client_sock, addr = server.accept()
                handler = threading.Thread(
                    target=self._process_stream_client, 
                    args=(client_sock, addr, verify_sig, anchor_enabled)
                )
                handler.daemon = True
                handler.start()
        finally:
            server.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Quantum Data Transport Layer")
    subparsers = parser.add_subparsers(dest="action", required=True)

    rx = subparsers.add_parser("ingress")
    rx.add_argument("--interface", type=str, default="0.0.0.0")
    rx.add_argument("--port", type=int, default=7790)
    rx.add_argument("--verify", action="store_true")
    rx.add_argument("--anchor", action="store_true")

    tx = subparsers.add_parser("egress")
    tx.add_argument("--target-host", type=str, required=True)
    tx.add_argument("--port", type=int, default=7790)
    tx.add_argument("--payload", type=str, required=True)

    args = parser.parse_args()
    streamer = QuantumDistributedStreamer()

    if args.action == "ingress":
        streamer.launch_ingress_node(interface=args.interface, port=args.port, verify_sig=args.verify, anchor_enabled=args.anchor)
    elif args.action == "egress":
        stamped_sig = hmac.new(streamer.secret_bytes, msg=args.payload.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
        transmission_headers = {"X-Payload-Signature-SHA256": stamped_sig}
        
        output_metrics = streamer.stream_package_to_remote(
            host=args.target_host,
            port=args.port,
            headers=transmission_headers,
            payload=args.payload
        )
        sys.stdout.write(json.dumps(output_metrics, indent=4) + "\n")
