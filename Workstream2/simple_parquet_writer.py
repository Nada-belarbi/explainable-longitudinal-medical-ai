"""Small dependency-free Parquet writer for the blocked MRI audit table.

Supports nullable UTF8 BYTE_ARRAY columns only, with PLAIN encoded values,
RLE encoded definition levels, uncompressed DataPage V1, and one row group.
Production list-valued features must use pyarrow in the fitted pipeline.
"""
import io
import struct
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from thrift.Thrift import TType


def emit_struct(p, fields):
    p.writeStructBegin('')
    for field_id, kind, val in fields:
        p.writeFieldBegin('', kind, field_id)
        if kind == TType.I32: p.writeI32(val)
        elif kind == TType.I64: p.writeI64(val)
        elif kind == TType.STRING: p.writeString(val)
        elif kind == TType.STRUCT: emit_struct(p, val)
        elif kind == TType.LIST:
            item_kind, items = val
            p.writeListBegin(item_kind, len(items))
            for item in items:
                if item_kind == TType.STRUCT: emit_struct(p, item)
                elif item_kind == TType.I32: p.writeI32(item)
                elif item_kind == TType.STRING: p.writeString(item)
                else: raise ValueError('unexpected list type')
            p.writeListEnd()
        else: raise ValueError('unexpected kind')
        p.writeFieldEnd()
    p.writeFieldStop()
    p.writeStructEnd()


def encode_thrift(fields):
    buf = TTransport.TMemoryBuffer()
    p = TCompactProtocol.TCompactProtocol(buf)
    emit_struct(p, fields)
    return buf.getvalue()


def uintvar(n):
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def def_levels(values):
    runs = []
    current = None
    count = 0
    for x in values:
        valid = 0 if x is None else 1
        if current is None: current, count = valid, 1
        elif current == valid: count += 1
        else:
            runs.extend([uintvar(count << 1), bytes([current])])
            current, count = valid, 1
    if count: runs.extend([uintvar(count << 1), bytes([current])])
    out = b''.join(runs)
    return struct.pack('<I', len(out)) + out


def write_parquet(path, columns, metadata=None):
    """columns: dict[str, list[str | None]]. All columns nullable UTF8."""
    names = list(columns)
    if not names: raise ValueError('No columns')
    n = len(columns[names[0]])
    if n == 0 or any(len(v) != n for v in columns.values()):
        raise ValueError('Missing or inconsistent rows')
    body = io.BytesIO()
    body.write(b'PAR1')
    chunks = []
    for name, values in columns.items():
        encoded = b''.join(struct.pack('<I', len(str(v).encode('utf-8'))) + str(v).encode('utf-8') for v in values if v is not None)
        page_body = def_levels(values) + encoded
        page_header = encode_thrift([
            (1, TType.I32, 0), # data page
            (2, TType.I32, len(page_body)),
            (3, TType.I32, len(page_body)),
            (5, TType.STRUCT, [
                (1,TType.I32,n),
                (2,TType.I32,0),  # PLAIN
                (3,TType.I32,3),  # RLE
                (4,TType.I32,3),
            ]),
        ])
        off = body.tell()
        body.write(page_header)
        body.write(page_body)
        length = len(page_header) + len(page_body)
        meta = [
            (1,TType.I32,6), # BYTE_ARRAY
            (2,TType.LIST,(TType.I32,[0,3])), # plain & rle
            (3,TType.LIST,(TType.STRING,[name])),
            (4,TType.I32,0), # uncompressed
            (5,TType.I64,n),
            (6,TType.I64,length),
            (7,TType.I64,length),
            (9,TType.I64,off),
        ]
        chunks.append([
            (2,TType.I64,off),
            (3,TType.STRUCT,meta),
        ])
    schema = [[(4,TType.STRING,'schema'), (5,TType.I32,len(names))]]
    for name in names:
        schema.append([
            (1,TType.I32,6), # BYTE_ARRAY
            (3,TType.I32,1), # OPTIONAL
            (4,TType.STRING,name),
            (6,TType.I32,0), # UTF8 converted type
        ])
    rowgroup = [
        (1,TType.LIST,(TType.STRUCT,chunks)),
        (2,TType.I64,body.tell()-4),
        (3,TType.I64,n),
    ]
    kv = []
    for k,v in (metadata or {}).items():
        kv.append([(1,TType.STRING,str(k)),(2,TType.STRING,str(v))])
    footer_fields = [
        (1,TType.I32,1),
        (2,TType.LIST,(TType.STRUCT,schema)),
        (3,TType.I64,n),
        (4,TType.LIST,(TType.STRUCT,[rowgroup])),
        (6,TType.STRING,'MRI Workstream 2 minimal Parquet writer'),
    ]
    if kv: footer_fields.insert(4,(5,TType.LIST,(TType.STRUCT,kv)))
    footer = encode_thrift(footer_fields)
    body.write(footer)
    body.write(struct.pack('<I',len(footer)))
    body.write(b'PAR1')
    with open(path,'wb') as f: f.write(body.getvalue())
    return {'rows':n,'columns':names,'bytes':body.tell()}
