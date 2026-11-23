"""
Tiny synthetic raw files for tests: write a handful of scans as mzML or mzXML so the real readers and stages can
run on them (no mocks). Scans are (ms_level, rt_sec, [(mz, intensity), ...], precursor_mz or None).
"""
import base64
import struct
from pathlib import Path


def _b64(values):
    return base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode()


def write_mzml(path: Path, scans) -> Path:
    out = []
    for i, (level, rt, peaks, prec) in enumerate(scans):
        mz = [p[0] for p in peaks]
        it = [p[1] for p in peaks]
        precursor = ""
        if prec is not None:
            precursor = (f'<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>'
                         f'<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="{prec}"/>'
                         f'</selectedIon></selectedIonList></precursor></precursorList>')
        out.append(f'''<spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="{len(peaks)}">
  <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{level}"/>
  <scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt / 60.0}" unitName="minute"/></scan></scanList>
  {precursor}
  <binaryDataArrayList count="2">
   <binaryDataArray encodedLength="0"><cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/><cvParam cvRef="MS" accession="MS:1000576" name="no compression"/><cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/><binary>{_b64(mz)}</binary></binaryDataArray>
   <binaryDataArray encodedLength="0"><cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/><cvParam cvRef="MS" accession="MS:1000576" name="no compression"/><cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/><binary>{_b64(it)}</binary></binaryDataArray>
  </binaryDataArrayList>
</spectrum>''')
    path.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><run id="r"><spectrumList count="{len(scans)}">
{chr(10).join(out)}
</spectrumList></run></mzML>''')
    return path


def write_mzxml(path: Path, scans) -> Path:
    out = []
    for i, (level, rt, peaks, prec) in enumerate(scans):
        flat = [v for p in peaks for v in p]
        raw = base64.b64encode(struct.pack(f">{len(flat)}d", *flat)).decode()
        precursor = f'<precursorMz precursorIntensity="1000">{prec}</precursorMz>' if prec is not None else ""
        out.append(f'<scan num="{i + 1}" msLevel="{level}" peaksCount="{len(peaks)}" retentionTime="PT{rt}S">'
                   f'{precursor}<peaks precision="64" byteOrder="network" contentType="m/z-int" '
                   f'compressionType="none" compressedLen="0">{raw}</peaks></scan>')
    path.write_text(f'''<?xml version="1.0" encoding="ISO-8859-1"?>
<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"><msRun scanCount="{len(scans)}">
{chr(10).join(out)}
</msRun></mzXML>''')
    return path


# MS1 with a pure target ion + a contaminant inside a 1.0 Da window, then an MS2 selecting the target.
# purity at the target (within +-0.01) = 800 / (800 + 200) = 0.8
SCANS = [
    (1, 1.0, [(100.0000, 800.0), (100.3000, 200.0), (250.0, 50.0)], None),
    (2, 1.2, [(50.0, 10.0), (75.0, 40.0)], 100.0),
    (1, 2.0, [(100.0000, 900.0), (100.3000, 100.0)], None),
    (2, 2.2, [(50.0, 5.0), (75.0, 60.0)], 100.0),
]
