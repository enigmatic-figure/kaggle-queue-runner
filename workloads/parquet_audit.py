"""CPU-only Parquet row/null/range statistics, with disk spill and a sealed result."""
from pathlib import Path
import argparse,json,os,tempfile
import duckdb
from queue_runner.sdk import output_dir,progress,checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--memory-gb',type=int,default=16)
    args=p.parse_args()
    if args.memory_gb<1:raise ValueError('Positive memory budget required')
    with tempfile.TemporaryDirectory(prefix='queue-duckdb-') as spill:
        con=duckdb.connect(config={'threads':int(os.environ.get('OMP_NUM_THREADS','1')),
            'memory_limit':str(args.memory_gb)+'GB','temp_directory':spill})
        try:
            con.from_parquet(args.input).create_view('audit_input')
            columns=con.execute('DESCRIBE audit_input').fetchall();count=con.execute('SELECT count(*) FROM audit_input').fetchone()[0]
            result={'rows':count,'input':args.input,'columns':[]}
            for name,kind,*_ in columns:
                column='"'+name.replace('"','""')+'"'
                fields='count(*)-count('+column+')'
                numeric=kind.split('(')[0] in ['TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT','UTINYINT','USMALLINT','UINTEGER','UBIGINT','FLOAT','DOUBLE','DECIMAL']
                if numeric:fields+=', min('+column+'), max('+column+'), avg('+column+'), stddev_pop('+column+')'
                values=con.execute('SELECT '+fields+' FROM audit_input').fetchone()
                record={'name':name,'type':kind,'nulls':values[0]}
                if numeric:record.update(zip(['min','max','mean','stddev'],values[1:]))
                result['columns'].append(record);progress(stage='column_audit',column=name,rows=count)
        finally:con.close()
    closed=output_dir()/'parquet-audit';closed.mkdir();(closed/'summary.json').write_text(json.dumps(result,indent=2,default=str));checkpoint('parquet-audit')


if __name__=='__main__':main()
