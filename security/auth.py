import hmac
import hashlib

def generate_signal_signature(secret_key: str, payload_str: str) -> str:
    return hmac.new(secret_key.encode(), payload_str.encode(), hashlib.sha256).hexdigest()

def verify_signal_signature(secret_key: str, payload_str: str, signature: str) -> bool:
    expected_signature = generate_signal_signature(secret_key, payload_str)
    return hmac.compare_digest(expected_signature, signature)

if __name__ == "__main__":
    key = "healthai-secret"
    data = "sensor_payload_data_string"
    sig = generate_signal_signature(key, data)
    print(f"[Security] Generated Payload Signature: {sig}")
    print(f"[Security] Signature Verification Status: {verify_signal_signature(key, data, sig)}")
