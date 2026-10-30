"""One versioned identity recipe for BOTH workstreams; no IDs in diagnostics."""
import hashlib
import hmac
import json


def subject_key(site: str, participant: str, secret: bytes) -> str:
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError('A new private secret of at least 32 bytes is required')
    blob_header = ('blob ' + str(len(secret)) + '\0').encode()
    if hashlib.sha1(blob_header + secret).hexdigest() == '13ce176f2ec3bef436b85f2d7f3ab2b0d53dc292':
        raise ValueError('The publicly tracked pseudonymization secret must be replaced')
    if any(not isinstance(v, str) or not v or v != v.strip() for v in (site, participant)):
        raise ValueError('Site and participant must be nonempty canonical strings')
    payload = json.dumps(['nacc74-site-participant-v1', site, participant],
                         ensure_ascii=False, separators=(',', ':')).encode('utf8')
    return 'n74_' + hmac.new(secret, payload, hashlib.sha256).hexdigest()
