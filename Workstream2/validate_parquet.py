"""Validate the basic Parquet metadata and 325-row UTF8 output without pyarrow.
This validates the writer's blocked export only. For complete compatibility use
an independent Parquet reader such as pyarrow after installation.
"""
import io, struct
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from thrift.Thrift import TType


def parse_struct(p):
    p.readStructBegin()
    ans={}
    while True:
        name,t,ident=p.readFieldBegin()
        if t==TType.STOP:break
        if t==TType.I32: v=p.readI32()
        elif t==TType.I64: v=p.readI64()
        elif t==TType.STRING: v=p.readString()
        elif t==TType.STRUCT: v=parse_struct(p)
        elif t==TType.LIST:
            el,nn=p.readListBegin()
            v=[]
            for _ in range(nn):
                if el==TType.STRUCT:v.append(parse_struct(p))
                elif el==TType.I32:v.append(p.readI32())
                elif el==TType.STRING:v.append(p.readString())
                else:raise ValueError('Unexpected field type')
            p.readListEnd()
        else:
            p.skip(t)
            v=None
        ans[ident]=v
        p.readFieldEnd()
    p.readStructEnd()
    return ans


def varint(data, pos):
    n=0; shift=0
    while True:
        b=data[pos];pos+=1
        n|=(b&127)<<shift
        if b<128:break
        shift+=7
    return n,pos


def read_column_bytes(data,meta,n):
    buf=TTransport.TMemoryBuffer(data[meta[9]:])
    p=TCompactProtocol.TCompactProtocol(buf)
    page=parse_struct(p)
    assert page[1]==0 and page[5][1]==n, 'Incorrect data page'
    pos=meta[9] + buf.cstringio_buf.tell()
    body=data[pos:pos+page[2]]
    assert len(body)==page[3], 'Bad body size'
    levels_length=struct.unpack('<I',body[:4])[0]
    level_bytes=body[4:4+levels_length]
    levels=[]; i=0
    while i<len(level_bytes):
        hdr,i=varint(level_bytes,i)
        assert hdr%2==0,'Unexpected bit packed levels'
        valid=level_bytes[i];i+=1
        levels.extend([valid]*(hdr>>1))
    assert len(levels)==n
    d=body[4+levels_length:]
    result=[];pos=0
    for level in levels:
        if level==0: result.append(None)
        else:
            size=struct.unpack_from('<I',d,pos)[0];pos+=4
            v=d[pos:pos+size].decode('utf-8');pos+=size
            result.append(v)
    assert pos==len(d), 'Page contained unparsed bytes'
    return result


def validate(path):
    data=open(path,'rb').read()
    assert data[:4]==b'PAR1' and data[-4:]==b'PAR1'
    size=struct.unpack('<I',data[-8:-4])[0]
    assert size>0 and size<len(data)
    buf=TTransport.TMemoryBuffer(data[-8-size:-8])
    meta=parse_struct(TCompactProtocol.TCompactProtocol(buf))
    n=meta[3]
    rows=meta[4]
    assert len(rows)==1
    assert rows[0][3]==n
    schema=[s[4] for s in meta[2][1:]]
    chunks=rows[0][1]
    assert len(schema)==len(chunks)
    result={}
    for name,chunk in zip(schema,chunks):
        assert chunk[3][3]==[name]
        result[name]=read_column_bytes(data,chunk[3],n)
    return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('file')
    args=ap.parse_args()
    data=validate(args.file)
    print('Parquet format checks passed',len(next(iter(data.values()))),'rows',len(data),'columns')
    print('matched_mri:', data['matched_mri'].count('true'))
    print('fit vectors present:',sum(x is not None for x in data['imaging_vector']))
