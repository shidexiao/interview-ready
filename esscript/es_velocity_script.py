import datetime
import dateutil.relativedelta
import time

start = "202401"
end = "202512"
ipAddress = "192.168.2.120:9200"


def createAlias(dateStartArray):
	indexAlias = ""

	indexAlias = indexAlias + str("{\"add\": {\"index\": \"fraud-" + dateStartArray.strftime("%Y%m"))
	indexAlias = indexAlias + str("\", \"alias\": \"fraud-r-" + dateStartArray.strftime("%Y%m"))

	indexAlias = indexAlias + str("\"}}, {\"add\": {\"index\": \"fraud-" + dateStartArray.strftime("%Y%m"))
	indexAlias = indexAlias + str("\", \"alias\": \"fraud-w-"+ dateStartArray.strftime("%Y%m") + "\"}},")
	
	return indexAlias

timeStart = time.strptime(start,"%Y%m")
timeEnd = time.strptime(end,"%Y%m")
timeStartStamp = int(time.mktime(timeStart))
timeEndStamp = int(time.mktime(timeEnd))
dateStartArray = datetime.datetime.fromtimestamp(timeStartStamp)
dateEndArray = datetime.datetime.fromtimestamp(timeEndStamp)

tagStart = str("curl -XPOST http://" + ipAddress + "/_aliases -d '{\"actions\": [")
tagEnd = str("{\"add\": {\"index\": \"fraud-other\", \"alias\": \"fraud-r-other\"}}, {\"add\": {\"index\": \"fraud-other\", \"alias\": \"fraud-w-other\"}}]}'" + " -u elastic:changeme" + " -H Content-Type:application/json" )
tagContent = ""

mapping = '{"settings":{"number_of_shards":3,"number_of_replicas":1,"index.mapping.total_fields.limit":100000},"mappings":{"history":{"_all":{"enabled":false},"date_detection":false,"dynamic":false,"dynamic_templates":[{"activity_doc_values":{"path_match":"activity.*","match_mapping_type":"string","mapping":{"fields":{"raw":{"ignore_above":256,"index":"not_analyzed","type":"keyword","doc_values":true}},"type":"text"}}},{"disable_doc_values":{"path_unmatch":"activity.*","match_mapping_type":"string","mapping":{"fields":{"raw":{"ignore_above":256,"index":"not_analyzed","type":"keyword","doc_values":false}},"type":"text"}}},{"long":{"match_mapping_type":"long","mapping":{"ignore_malformed":true,"type":"long"}}},{"double":{"match_mapping_type":"double","mapping":{"ignore_malformed":true,"type":"double"}}},{"boolean":{"match_mapping_type":"boolean","mapping":{"ignore_malformed":true,"type":"boolean"}}}],"properties":{"activity":{"dynamic":true,"properties":{"create":{"doc_values":"true","type":"long"}}},"address":{"dynamic":true,"properties":{}},"company":{"dynamic":true,"properties":{}}}}}}'

f = open("elasticsearch-fraud-init.sh","w")
while(dateStartArray <= dateEndArray):
	strPut = str("curl -XPUT http://" + ipAddress + "/fraud-" + dateStartArray.strftime("%Y%m")+" -d '" + mapping + "' -u elastic:changeme"  + " -H Content-Type:application/json")
	print >> f,strPut
	tagContent = tagContent + createAlias(dateStartArray)

	dateStartArray = dateStartArray + dateutil.relativedelta.relativedelta(months=1)

print >> f,str("curl -XPUT http://" + ipAddress + "/fraud-other -d '" + mapping + "' -u elastic:changeme"  + " -H Content-Type:application/json")
print >> f,str(tagStart + tagContent + tagEnd)
