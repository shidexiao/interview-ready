import datetime
import time

#startdate should be more 1,exp:i want 0101,so i input 0102,the result would run out from 0101
start = "20241001"
end = "20251231"
ipAddress = "192.168.2.120:9200"

def createAlias(dateStartArray):
	indexAlias = ""

	i = 0;
	while(i < 7):
		everyDay = dateStartArray + datetime.timedelta(days = i)
		indexAlias = indexAlias + str("{\"add\": {\"index\": \"freyr-" + dateStartArray.strftime("%Y%m%d"))
		indexAlias = indexAlias + str("\", \"alias\": \"freyr-r-" + everyDay.strftime("%Y%m%d"))
		indexAlias = indexAlias + str("\"}}, {\"add\": {\"index\": \"freyr-" + dateStartArray.strftime("%Y%m%d"))
		indexAlias = indexAlias + str("\", \"alias\": \"freyr-w-"+ everyDay.strftime("%Y%m%d") + "\"}},")
		i = i + 1

	return indexAlias

timeStart = time.strptime(start,"%Y%m%d")
timeEnd = time.strptime(end,"%Y%m%d")
timeStartStamp = int(time.mktime(timeStart))
timeEndStamp = int(time.mktime(timeEnd))
dateStartArray = datetime.datetime.utcfromtimestamp(timeStartStamp)
dateEndArray = datetime.datetime.utcfromtimestamp(timeEndStamp)

tagStart = str("curl -XPOST http://" + ipAddress + "/_aliases -d '{\"actions\": [")
tagEnd = str("{\"add\": {\"index\": \"freyr-other\", \"alias\": \"freyr-r-other\"}}, {\"add\": {\"index\": \"freyr-other\", \"alias\": \"freyr-w-other\"}}]}'" + " -u elastic:changeme" + " -H Content-Type:application/json" )
tagContent = ""

f = open("elasticsearch-init-freyr.sh","w")
while(dateStartArray < dateEndArray):
	strPut = str("curl -XPUT http://" + ipAddress + "/freyr-"+dateStartArray.strftime("%Y%m%d")+" -d '{\"mappings\":{\"_default_\":{\"_all\": {\"enabled\": false},\"date_detection\":false,\"dynamic_templates\":[{\"str\":{\"match_mapping_type\":\"string\",\"mapping\":{\"fields\":{\"raw\":{\"type\":\"keyword\",\"ignore_above\":256}},\"type\":\"text\",\"norms\": false}}}]},\"serviceInvoke\":{\"properties\":{\"responseData.data\":{\"index\":\"false\",\"type\":\"text\",\"norms\": false},\"processData.data\":{\"index\":\"false\",\"type\":\"text\",\"norms\": false},\"responseData.processData\":{\"index\":\"false\",\"type\":\"text\",\"norms\": false}}}},\"settings\":{\"number_of_replicas\":1,\"number_of_shards\":3}}'" + " -u elastic:changeme" + " -H Content-Type:application/json" )
	print >> f,strPut
	tagContent = tagContent + createAlias(dateStartArray)
	dateStartArray = dateStartArray + datetime.timedelta(days = 7)

print >> f,str("curl -XPUT http://" + ipAddress + "/freyr-other -d '{\"mappings\":{\"_default_\":{\"_all\": {\"enabled\": false},\"date_detection\":false,\"dynamic_templates\":[{\"str\":{\"match_mapping_type\":\"string\",\"mapping\":{\"fields\":{\"raw\":{\"type\":\"keyword\",\"ignore_above\":256}},\"type\":\"text\",\"norms\": false}}}]},\"serviceInvoke\":{\"properties\":{\"responseData.data\":{\"index\":\"false\",\"type\":\"text\",\"norms\": false},\"processData.data\":{\"index\":\"false\",\"type\":\"text\",\"norms\": false},\"responseData.processData\":{\"index\":\"false\",\"type\":\"text\",\"norms\": false}}}},\"settings\":{\"number_of_replicas\":1,\"number_of_shards\":3}}'" + " -u elastic:changeme" + " -H Content-Type:application/json" )
print >> f,str(tagStart + tagContent + tagEnd)
