import datetime
import time

#startdate should be more 1,exp:i want 0101,so i input 0102,the result would run out from 0101
start = "20241001"
end = "20251231"
ipAddress = "192.168.2.120:9200"
interval = 30

def createAlias(dateStartArray):
	indexAlias = ""

	i = 0;
	while(i < interval):
		everyDay = dateStartArray + datetime.timedelta(days = i)
		indexAlias = indexAlias + str("{\"add\": {\"index\": \"atreus_componentlog-" + dateStartArray.strftime("%Y%m%d"))
		indexAlias = indexAlias + str("\", \"alias\": \"atreus_componentlog-r-" + everyDay.strftime("%Y%m%d"))
		indexAlias = indexAlias + str("\"}}, {\"add\": {\"index\": \"atreus_componentlog-" + dateStartArray.strftime("%Y%m%d"))
		indexAlias = indexAlias + str("\", \"alias\": \"atreus_componentlog-w-"+ everyDay.strftime("%Y%m%d") + "\"}},")
		i = i + 1

	return indexAlias

timeStart = time.strptime(start,"%Y%m%d")
timeEnd = time.strptime(end,"%Y%m%d")
timeStartStamp = int(time.mktime(timeStart))
timeEndStamp = int(time.mktime(timeEnd))
dateStartArray = datetime.datetime.utcfromtimestamp(timeStartStamp)
dateEndArray = datetime.datetime.utcfromtimestamp(timeEndStamp)

tagStart = str("curl -XPOST http://" + ipAddress + "/_aliases -d '{\"actions\": [")
tagEnd = str("{\"add\": {\"index\": \"atreus_componentlog-other\", \"alias\": \"atreus_componentlog-r-other\"}}, {\"add\": {\"index\": \"atreus_componentlog-other\", \"alias\": \"atreus_componentlog-w-other\"}}]}'" + " -u elastic:changeme"  + " -H Content-Type:application/json")
tagContent = ""

f = open("elasticsearch-componentlog-init2.sh","w")
while(dateStartArray < dateEndArray):
	strPut = str("curl -XPUT http://" + ipAddress + "/atreus_componentlog-"+dateStartArray.strftime("%Y%m%d")+" -d '{\"mappings\":{\"_default_\":{\"_all\":{\"enabled\":false},\"dynamic\":false,\"date_detection\":false},\"componentlog\":{\"_all\":{\"enabled\":false},\"dynamic\":false,\"date_detection\":false,\"properties\":{\"token\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"statDate\":{\"type\":\"long\",\"doc_values\":false,\"ignore_malformed\":true},\"nodeType\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"nodeId\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"nodeName\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}}}}},\"settings\":{\"number_of_replicas\":1,\"number_of_shards\":3}}'" + " -u elastic:changeme" + " -H Content-Type:application/json")
	print(strPut,file=f)
	tagContent = tagContent + createAlias(dateStartArray)
	dateStartArray = dateStartArray + datetime.timedelta(days = interval)

# print >> f,str("curl -XPUT http://" + ipAddress + "/atreus_componentlog-other -d '{\"mappings\":{\"_default_\":{\"_all\":{\"enabled\":false},\"dynamic\":false,\"date_detection\":false},\"componentlog\":{\"_all\":{\"enabled\":false},\"dynamic\":false,\"date_detection\":false,\"properties\":{\"token\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"statDate\":{\"type\":\"long\",\"doc_values\":false,\"ignore_malformed\":true},\"nodeType\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"nodeId\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"nodeName\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}}}}},\"settings\":{\"number_of_replicas\":1,\"number_of_shards\":3}}'" + " -u elastic:changeme" + " -H Content-Type:application/json")
# print >> f,str(tagStart + tagContent + tagEnd)

print(str("curl -XPUT http://" + ipAddress + "/atreus_componentlog-other -d '{\"mappings\":{\"_default_\":{\"_all\":{\"enabled\":false},\"dynamic\":false,\"date_detection\":false},\"componentlog\":{\"_all\":{\"enabled\":false},\"dynamic\":false,\"date_detection\":false,\"properties\":{\"token\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"statDate\":{\"type\":\"long\",\"doc_values\":false,\"ignore_malformed\":true},\"nodeType\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"nodeId\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}},\"nodeName\":{\"type\":\"text\",\"doc_values\":false,\"norms\":false,\"fields\":{\"raw\":{\"type\":\"keyword\",\"doc_values\":false,\"ignore_above\":256}}}}}},\"settings\":{\"number_of_replicas\":1,\"number_of_shards\":3}}'" + " -u elastic:changeme" + " -H Content-Type:application/json"),file=f)
print(str(tagStart + tagContent + tagEnd),file=f)