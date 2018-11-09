file_dir=/data/out14
ck_conn=localhost
dis_table=dis_event
for file_path in `ls $file_dir | grep 'sort' `;do
	echo "load  ${file_dir}/${file_path}"
      ../importer -dsn=tcp://${ck_conn}:9000  -file=${file_dir}/${file_path} -logAll=false -table=${dis_table} -pktype=String
done
