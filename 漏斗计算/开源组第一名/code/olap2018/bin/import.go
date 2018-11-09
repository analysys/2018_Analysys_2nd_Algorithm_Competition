package main

import (
	"bufio"
	"bytes"
	"database/sql"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/housepower/olap2018/util"

	json "github.com/json-iterator/go"
	_ "github.com/kshvakov/clickhouse"
	"github.com/satori/go.uuid"
)

//这个脚本处理单个文件的数据, 导入到ck中
// usage:
// go run bin/import.go -file=/data/yiguan/dataset/test.data
var (
	file string
	lock sync.Mutex

	emptyjs    = []byte("{}")
	prepareSQL string

	strKeys   = []string{"city", "name", "brand"}
	floatKeys = []string{"price"}
	intKeys   = []string{"nums", "how"}

	SEP      = []byte("\t")
	DATE_SEP = []byte("-")
	COL_SIZE = 11

	BATCH_SIZE = 81920

	cols = "uid, its, action_code, action_name, event_city, event_name, event_brand, event_price, event_nums, event_how, day"

	//flag
	dsn          = fmt.Sprintf("tcp://%s", "172.27.138.8:9000")
	writeThreads = 32

	logAll  bool
	table   string
	pk_type string
)

func init() {
	flag.StringVar(&file, "file", "", "file to load")
	flag.StringVar(&pk_type, "pktype", "String", "String,UUID")

	flag.BoolVar(&logAll, "logAll", true, "shoule log the process")
	flag.StringVar(&table, "table", "dis_event", "table")
	flag.StringVar(&dsn, "dsn", fmt.Sprintf("tcp://%s", "localhost:9000"), "clickhouse conn")

	flag.Parse()

	var params = make([]string, COL_SIZE)
	for i := range params {
		params[i] = "?"
	}

	prepareSQL = "INSERT INTO " + table + " (" + cols + ") VALUES (" + strings.Join(params, ",") + ")"

	logs("SQL=>", prepareSQL)
}

func main() {
	proc := NewProcessor(file, dsn, writeThreads)
	proc.Run()
	proc.Wait()
}

type Processor struct {
	fp   string
	conn *sql.DB

	//写入并发数
	writeThreads int
	writePool    chan struct{}
	wg           sync.WaitGroup

	dataC []chan string
	idx   int
}

func NewProcessor(fp string, dsn string, writeThreads int) *Processor {
	sqlDB, err := sql.Open("clickhouse", dsn)
	if err != nil {
		log.Fatal(err)
	}

	writePool := make(chan struct{}, writeThreads)
	var wg sync.WaitGroup

	dataC := make([]chan string, writeThreads)
	for i := range dataC {
		dataC[i] = make(chan string, 100000)
	}

	return &Processor{
		fp, sqlDB, writeThreads, writePool, wg, dataC, 0,
	}
}

func (p *Processor) Run() {
	go p.startRead()
}

// sample data : 9397688616950152284 1529454007 searchGoods 搜索商品 {"name":"watch","city":"深圳","brand":"Hair","price":6034.851} 20180620
// 数据为文本文件格式，具体包含字段有：
// (1）用户ID，字符串类型
// (2）时间戳，秒级别，Long类型
// (3）事件CODE，字符串类型，包含startUp、login、searchGoods等15个事件
// (4）事件名称，字符串类型，包含启动、登陆、搜索商品等15个事件
// (5）事件属性，Json串格式 。包含，city：字符串；name:字符串；brand:字符串；price:浮点型（3位精度），nums：整型,how：整型；
// (6）日期，字符串类型
// 324533120  324533120
// 测试数据总条数3亿左右，日期范围：2018/06/01到2018/07/05。
// 比赛数据总条数10亿左右, 日期范围：2018/06/01到2018/07/15。
func (p *Processor) startRead() {
	r, err := os.Open(p.fp)
	if err != nil {
		panic(err)
	}
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		p.push(sc.Text())
	}

	//开始等待
	for i := range p.dataC {
		close(p.dataC[i])
	}
}

func (p *Processor) push(str string) {
	p.idx = (p.idx + 1) % p.writeThreads
	p.dataC[p.idx] <- str
}

func (p *Processor) startIngest(idx int) {
	var metricBatches = make([]*Metric, 0, BATCH_SIZE)

	var i = 0

	for str := range p.dataC[idx] {
		i = i + 1
		metric := NewMetric(COL_SIZE)
		dataBs := util.String2Bytes(str)
		bs := bytes.Split(dataBs, SEP)

		// uid
		if pk_type == "UUID" {
			a := string(bs[0])
			b := strings.Replace(a, "-", "", -1)
			h8, err := strconv.ParseInt(string(b[:16]), 10, 64)

			if err != nil {
				log.Fatalf("error in h8 %s:%s   %s", b, a, err.Error())
				panic(err)
			}

			l8, err := strconv.ParseInt(string(b[16:]), 10, 64)
			if err != nil {
				log.Fatalf("error in l8  %s:%s  %s", b, a, err.Error())
				panic(err)
			}

			if h8+l8 == 0 {
				log.Fatalf("error in result %s", b)
			}

			var u uuid.UUID
			binary.LittleEndian.PutUint64(u[:8], uint64(h8))
			binary.LittleEndian.PutUint64(u[8:], uint64(l8))
			metric.WriteBytes(u.Bytes())
		} else {
			metric.WriteString(bs[0])
		}

		// metric.WriteString(bs[0])

		//timestamp
		metric.WriteInt(bs[1])
		//事件CODE
		metric.WriteString(bs[2])
		//事件名称
		metric.WriteString(bs[3])

		it := make(map[string]interface{})
		if !bytes.Equal(bs[4], emptyjs) {
			err := json.Unmarshal(bs[4], &it)
			if err != nil {
				log.Fatal(string(bs[4]), "of", str)
				panic(err.Error())
			}
		}

		for _, key := range strKeys {
			if v, ok := it[key]; ok {
				metric.WriteVal(v.(string))
			} else {
				metric.WriteVal("")
			}
		}

		for _, key := range floatKeys {
			if v, ok := it[key]; ok {
				metric.WriteVal(v.(float64))
			} else {
				metric.WriteVal(float64(0.0))
			}
		}

		for _, key := range intKeys {
			if v, ok := it[key]; ok {
				metric.WriteVal(int64(v.(float64)))
			} else {
				metric.WriteVal(0)
			}
		}

		//转date类型 20160707 => 2016-07-07
		dd := bs[5]
		metric.WriteString(bytes.Join([][]byte{dd[:4], dd[4:6], dd[6:8]}, DATE_SEP))
		metricBatches = append(metricBatches, metric)

		if len(metricBatches) >= BATCH_SIZE {
			p.flush(metricBatches)
			metricBatches = make([]*Metric, 0, BATCH_SIZE)
		}
	}
	if len(metricBatches) > 0 {
		p.flush(metricBatches)
	}
	if logAll {
		log.Println("done ", idx, "records => ", i)
	}
}

func (p *Processor) flush(metricBatches []*Metric) {
	p.wg.Add(1)
	p.writePool <- struct{}{}
	go func() {
		p.write(metricBatches)
		<-p.writePool
		p.wg.Done()
	}()
}

func (p *Processor) Wait() {
	var ingestWg sync.WaitGroup
	for i := range p.dataC {
		ingestWg.Add(1)
		go func(i int) {
			p.startIngest(i)
			ingestWg.Done()
		}(i)
	}
	ingestWg.Wait()
	p.wg.Wait()
}

// 将数据插入到ck
func (p *Processor) write(metricBatches []*Metric) {
	tx, err := p.conn.Begin()
	if err != nil {
		log.Printf("clickhouse begin error %v \n", err.Error())
		return
	}

	stmt, err := tx.Prepare(prepareSQL)
	if err != nil {
		log.Printf("clickhouse prepare error %v \n", err.Error())
		return
	}

	for _, metric := range metricBatches {
		if _, err := stmt.Exec(metric.vals...); err != nil {
			log.Printf("clickhouse exec run err for %v \n", err.Error())
		}
	}

	if err := tx.Commit(); err != nil {
		log.Printf("clickhouse commit error %v \n", err.Error())
	}
	logs("batches insert success ==>", len(metricBatches))
}

type Metric struct {
	vals []interface{}
}

func NewMetric(n int) *Metric {
	return &Metric{vals: make([]interface{}, 0, n)}
}
func (m *Metric) WriteInt(data []byte) {
	str := util.Bytes2String(data)
	i, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		log.Printf("Parse int error %#v \n", str)
	}
	m.vals = append(m.vals, i)
}

func (m *Metric) WriteBytes(data []byte) {
	m.vals = append(m.vals, data)
}

func (m *Metric) WriteString(data []byte) {
	m.vals = append(m.vals, string(data))
}

func (m *Metric) WriteFloat(data []byte) {
	str := util.Bytes2String(data)
	f, err := strconv.ParseFloat(str, 32)
	if err != nil {
		log.Printf("Parse float error %#v \n", str)
	}
	m.vals = append(m.vals, f)
}

func (m *Metric) WriteVal(data interface{}) {
	m.vals = append(m.vals, data)
}

func logs(v ...interface{}) {
	if logAll {
		log.Println(v...)
	}
}
