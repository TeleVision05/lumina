/**
 * Read ModelProto.metadata_props from ONNX bytes without a protobuf library, so the app can tell
 * the visitor when the bundled model is the placeholder (metadata lumina.placeholder = "true").
 *
 * Wire format: ModelProto field 14 is `repeated StringStringEntryProto metadata_props`, whose
 * fields are 1 (key) and 2 (value), both strings. Every other top-level field is skipped by wire
 * type, so this only walks the top level and never decodes the graph.
 */

class Reader {
  pos = 0;
  constructor(private readonly buf: Uint8Array) {}

  get done(): boolean {
    return this.pos >= this.buf.length;
  }

  varint(): number {
    let result = 0;
    let scale = 1;
    for (let i = 0; i < 10; i++) {
      const b = this.buf[this.pos++];
      if (b === undefined) throw new RangeError('truncated varint');
      result += (b & 0x7f) * scale;
      if ((b & 0x80) === 0) return result;
      scale *= 128;
    }
    throw new RangeError('varint too long');
  }

  bytes(): Uint8Array {
    const len = this.varint();
    if (this.pos + len > this.buf.length) throw new RangeError('truncated field');
    const out = this.buf.subarray(this.pos, this.pos + len);
    this.pos += len;
    return out;
  }

  skip(wireType: number): void {
    if (wireType === 0) this.varint();
    else if (wireType === 1) this.pos += 8;
    else if (wireType === 2) this.bytes();
    else if (wireType === 5) this.pos += 4;
    else throw new RangeError(`unsupported wire type ${wireType}`);
  }
}

const decoder = new TextDecoder();

function readEntry(buf: Uint8Array): [string, string] {
  const r = new Reader(buf);
  let key = '';
  let value = '';
  while (!r.done) {
    const tag = r.varint();
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (field === 1 && wire === 2) key = decoder.decode(r.bytes());
    else if (field === 2 && wire === 2) value = decoder.decode(r.bytes());
    else r.skip(wire);
  }
  return [key, value];
}

export function readOnnxMetadata(model: Uint8Array): Record<string, string> {
  const r = new Reader(model);
  const out: Record<string, string> = {};
  while (!r.done) {
    const tag = r.varint();
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (field === 14 && wire === 2) {
      const [k, v] = readEntry(r.bytes());
      out[k] = v;
    } else {
      r.skip(wire);
    }
  }
  return out;
}
