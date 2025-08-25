-- Query查询结果默认限制200行，如需更多数据请修改limit，最多展示10000行或20M。
select * from mongo_ods.renhang_req order by create_time DESC limit 10;
select * from mongo_ods.renhang order by create_time DESC limit 10;


select * from mongo_ods.check_user_query_records;

select * from mongo_ods.check_user_query_records order by create_time DESC limit 10;

select * from mongo_ods.renhang where data_src='huarui';



select * from mongo_ods.qianhai_xiaoniu_scorea2 limit 10;

select * from mongo_ods.pudao_tx_fraud_v75 order by create_time DESC limit 10;


select * from mongo_ods.fl_finplus_largeloan limit 10;

select * from realtime_dws.dws_risk_idno_overdue_d_v limit 10;
select * from realtime_dws.dws_risk_idno_overdue_d_v limit 10

select * from zl_fk_data.risk_dz_adjust_reloan_var_recall_1


select
	problem_source	as zlkf_coi_problem_source--问题来源
	,channel as zlkf_coi_channel	--渠道方 1-现有API,2-上海接入流量渠道
	,fund_code as zlkf_coi_fund_code	--资金方
	,pro_first_catagory as zlkf_coi_pro_first_catagory		--问题一级分类
	,pro_second_catagory as zlkf_coi_pro_second_catagory	--问题二级分类
	,order_type	as zlkf_coi_order_type	--工单类型 1-贷后工单, 2-客服工单
	,order_status	as zlkf_coi_order_status --工单状态
	,follow_result as zlkf_coi_follow_result	--最近一次跟进结果
	,user_name as zlkf_coi_user_name	--客户姓名
	,mobile	as zlkf_coi_mobile	--来电号码
	,register_phone	 as zlkf_coi_register_phone	--注册号码
	,customer_no as zlkf_coi_customer_no	--客户号
	,idcard_no as zlkf_coi_idcard_no	--身份证号码
	,level as zlkf_coi_level 	--等级 1-去电, 2-咨询, 3-普通投诉, 4-重点投诉
	,loan_no as zlkf_coi_loan_no	--借据号
	,hander_result as zlkf_coi_hander_result	--处理结果
	,emergency_degree as zlkf_coi_emergency_degree	--紧急程度 1-较低, 2-普通, 3-紧急, 4-非常紧急
	,to_collect_remark as zlkf_coi_to_collect_remark	--移交催收备注
	,feedback_time as zlkf_coi_feedback_time	--反馈时间
	,create_time as zlkf_coi_create_time	--创建时间
FROM zlkf.customer_order_info
WHERE md5(idcard_no) = 'xxx' OR md5(mobile) = 'xxx' OR md5(register_phone) = 'xxx'
ORDER BY level DESC
LIMIT 1;


select * from zlkf.customer_order_info;



select
	problem_source	as zlkf_coi_problem_source--问题来源
	,channel as zlkf_coi_channel	--渠道方 1-现有API,2-上海接入流量渠道
	,fund_code as zlkf_coi_fund_code	--资金方
	,pro_first_catagory as zlkf_coi_pro_first_catagory		--问题一级分类
	,pro_second_catagory as zlkf_coi_pro_second_catagory	--问题二级分类
	,order_type	as zlkf_coi_order_type	--工单类型 1-贷后工单, 2-客服工单
	,order_status	as zlkf_coi_order_status --工单状态
	,follow_result as zlkf_coi_follow_result	--最近一次跟进结果
	,user_name as zlkf_coi_user_name	--客户姓名
	,mobile	as zlkf_coi_mobile	--来电号码
	,register_phone	 as zlkf_coi_register_phone	--注册号码
	,customer_no as zlkf_coi_customer_no	--客户号
	,idcard_no as zlkf_coi_idcard_no	--身份证号码
	,level as zlkf_coi_level 	--等级 1-去电, 2-咨询, 3-普通投诉, 4-重点投诉
	,loan_no as zlkf_coi_loan_no	--借据号
	,hander_result as zlkf_coi_hander_result	--处理结果
	,emergency_degree as zlkf_coi_emergency_degree	--紧急程度 1-较低, 2-普通, 3-紧急, 4-非常紧急
	,to_collect_remark as zlkf_coi_to_collect_remark	--移交催收备注
	,feedback_time as zlkf_coi_feedback_time	--反馈时间
	,create_time as zlkf_coi_create_time	--创建时间
FROM zlkf.customer_order_info
WHERE md5(idcard_no) = 'af78bbaee86d52c7c91344e8043f72f8' OR md5(mobile) = 'de4933f0f1e128646453f4412abe2a9f' OR md5(register_phone) = 'de4933f0f1e128646453f4412abe2a9f'
ORDER BY level DESC
LIMIT 1;

select * from mongo_ods.huaruishreportlog limit 10;

select * from mongo_ods.hongfeireportlog limit 10;

select * from mongo_ods.huaruishreportlog order by createtime DESC limit 10;

select * from mongo_ods.hongfeireportlog order by createtime DESC limit 10;
select * from mongo_ods.hongfeireportlog order by createtime ASC limit 10;

select count(*) from mongo_ods.hongfeireportlog;
select * from mongo_ods.baihang_dongzhen;

select * from mongo_ods.fl_3factor_upg

select count(*) from mongo_ods.fulin_online_status_coll where create_time >'2024-12-01' and  create_time <'2025-01-01';
select count(*) from mongo_ods.fulin_online_status_coll where create_time >'2024-12-01' and  create_time <'2025-01-01';

bairong_special_list
select count(*) from mongo_ods.fulin_online_status_coll where create_time >'2024-12-01' and  create_time <'2025-01-01';
select count(*) from mongo_ods.bairong_special_list where create_time >'2024-11-01' and  create_time <'2024-12-01';


SELECT  t1._id::text as id1
        ,t1.req_id
        ,t1.status :: text as status
        ,t2._id::text as id2
        ,t2.is_cache
        ,t3._id::text  as id3
        ,t3.risk_business_id::text  as risk_business_id
FROM    mongo_ods.fulin_online_status_coll t1
LEFT JOIN mongo_ods.fulin_online_status_map t2
ON      t1.req_id = t2.fulin_online_status_req_id
LEFT JOIN mongo_ods.fulin_online_status_req t3
ON      t2.customer_request_id = t3.customer_request_id
where date(t1.create_time)>='2025-01-01' and t3._id is null

select * from mongo_ods.fulin_online_status_coll where req_id in ('884e1995-3585-4526-b2f8-0ca16a92fc44','d2f91d80-cfd2-4441-b5d9-8906dbaccaef','1d1ab9dc-b7d1-486e-b15d-eb9f3504e7d3','5645e4a1-88ef-413d-9516-eca0e1c6e254','b0f33361-74f2-47b3-b027-1a1f572d7898')

select * from mongo_ods.fulin_online_status_map where fulin_online_status_req_id in ('884e1995-3585-4526-b2f8-0ca16a92fc44','d2f91d80-cfd2-4441-b5d9-8906dbaccaef','1d1ab9dc-b7d1-486e-b15d-eb9f3504e7d3','5645e4a1-88ef-413d-9516-eca0e1c6e254','b0f33361-74f2-47b3-b027-1a1f572d7898')


SELECT  t1._id::text as id1
        ,t1.req_id
        ,t1.status :: text as status
        ,t2._id::text as id2
        ,t2.is_cache
        ,t3._id::text  as id3
        ,t3.risk_business_id::text  as risk_business_id
FROM    mongo_ods.bairong_special_list t1
LEFT JOIN mongo_ods.bairong_special_list_map t2
ON      t1.req_id = t2.req_id
LEFT JOIN mongo_ods.bairong_special_list_req t3
ON      t2.customer_request_id = t3.customer_request_id
where date(t1.create_time)>='2025-01-01' and t3._id is null


SELECT  t1._id::text as id1
        ,t1.req_id
        ,t1.status :: text as status
        ,t2._id::text as id2
        ,t2.is_cache
        ,t3._id::text  as id3
        ,t3.risk_business_id::text  as risk_business_id
FROM    mongo_ods.bairong_special_list t1
LEFT JOIN mongo_ods.bairong_special_list_map t2
ON      t1.req_id = t2.req_id
LEFT JOIN mongo_ods.bairong_special_list_req t3
ON      t2.customer_request_id = t3.customer_request_id
where date(t1.create_time)>='2024-11-01' and t3._id is null

select * from mongo_ods.bairong_special_list where req_id in ('789062c4-c43d-47ad-8cfc-c969190d7e60','4823b8d6-63ea-4af0-833f-3aee7269f2b0','eccdc062-a507-4e07-b19d-1227d0b5e54d','68a3bd15-d66a-4491-a1e7-ada006e4795c','4ba1c9e4-eac6-4953-aa21-8bd375f6d649','d094244e-b7a9-4f69-9203-010e0aa53c6a','e4f7e06f-45c8-430c-9c06-abd7292c9cb7','31aa408b-18bd-4033-b237-452cd1bc251a','ba5cf569-b709-4335-ab25-4c40178af063','90daefb1-9396-44bd-836c-97c67fb4e2e6','057ec585-1684-43d4-b0e3-92aff44a44ae','25a5d46d-4202-4203-ae3e-169edc476b83','79c93393-d7ac-45fb-aae0-8979f0cfe943')


select result->>'bairong_sl_customer_request_id' from mongo_ods.bairong_special_list where req_id in ('789062c4-c43d-47ad-8cfc-c969190d7e60','4823b8d6-63ea-4af0-833f-3aee7269f2b0','eccdc062-a507-4e07-b19d-1227d0b5e54d','68a3bd15-d66a-4491-a1e7-ada006e4795c','4ba1c9e4-eac6-4953-aa21-8bd375f6d649','d094244e-b7a9-4f69-9203-010e0aa53c6a','e4f7e06f-45c8-430c-9c06-abd7292c9cb7','31aa408b-18bd-4033-b237-452cd1bc251a','ba5cf569-b709-4335-ab25-4c40178af063','90daefb1-9396-44bd-836c-97c67fb4e2e6','057ec585-1684-43d4-b0e3-92aff44a44ae','25a5d46d-4202-4203-ae3e-169edc476b83','79c93393-d7ac-45fb-aae0-8979f0cfe943')


SELECT  t1._id::text as id1
        ,t1.req_id
        ,t1.status :: text as status
        ,t2._id::text as id2
        ,t2.is_cache
        ,t3._id::text  as id3
        ,t3.risk_business_id::text  as risk_business_id
FROM    mongo_ods.bairong_hhc_level t1
LEFT JOIN mongo_ods.bairong_hhc_level_map t2
ON      t1.req_id = t2.req_id
LEFT JOIN mongo_ods.bairong_hhc_level_req t3
ON      t2.customer_request_id = t3.customer_request_id
where date(t1.create_time)>='2025-01-01' and t3._id is null


select  count(*) from     mongo_ods.baihang_dongzhen
where  date(create_time) >='2024-12-01' and date(create_time) <'2025-01-01' and status='2'

select * from atreus_es.freyr_20250113
select count(*) from atreus_es.freyr_partd_202504;
select count(*) from atreus_es.freyr_partd_202503;


select * from atreus_es."atreus_entryinvokeresult-20241229";
atreus_entryinvokeresult_20241229
select * from atreus_es."atreus_entryinvokeresult_20241229";

select * from atreus_es.atreus_entryinvokeresult_20241229;

select * from mongo_ods.query_risk;
select * from mongo_ods.query_risk_req;
select * from mongo_ods.query_risk_map;


select * from mongo_ods.tanzhi_thfd where _id = '677f77cfc5bdccc952ef04d1'

select * from mongo_ods.yinlian_zhice;
select * from mongo_ods.yinlian_zhice_req;
select * from mongo_ods.yinlian_zhice_map;

select * from mongo_ods.tanzhi_qyfxzs_req where risk_business_id='1888806859494952960';
select * from mongo_ods.tanzhi_qyfxzs_map where customer_request_id='e75e0a41-7791-4a2e-8d57-3cd2c9300525';
select * from mongo_ods.tanzhi_qyfxzs where req_id='24804476-5971-4c66-b706-232cccd1ca8c';


SELECT * FROM atreus_es.atreus_entryinvokeresult_20241229_tmp;


SELECT * FROM atreus_es.atreus_componentlog_20241229_tmp;



SELECT * FROM atreus_es."atreus_entryinvokeresult-20250826";

select * from mongo_ods.pudao_dhb_sjf_f01;



SELECT EXTRACT(EPOCH FROM TIMESTAMP '2020-02-27 00:11:00');
SELECT EXTRACT(EPOCH FROM TIMESTAMP '2020-02-27 00:11:00.123456') * 1000;
SELECT EXTRACT(EPOCH FROM TIMESTAMP '2020-02-27 00:11:00') * 1000;

select * from mongo_ods.fulin_online_status_req order by create_time DESC;

select * from mongo_ods.tengrui_r904_req;
select * from mongo_ods.tengrui_r904;



SELECT * FROM realtime.gio_event_message;


select * from zws_middleware.zx_credit_info;

select * from realtime.gio_event_message;
// 这个sink表结构怎么定义？ anonymous_id
// sdk传递的user_id
//


SELECT * FROM zws_middleware.zx_repayment_apply_record as a LEFT JOIN zws_middleware.zx_credit_info as b ON a.user_id=b.user_id;

begin;
call set_table_property('zws_middleware.zx_repayment_apply_record', 'binlog.level', 'replica');
commit;

begin;
call set_table_property('zws_middleware.zx_repayment_apply_record', 'binlog.ttl', '259200');
commit;
select * from realtime.gio_event_message;


-- gio --
-- 8	userCollisionRequestResult	用户撞库结果

SELECT * FROM zws_middleware.zl_decision_info_log;
SELECT final_deal_type_code,channel_code,refuse_msg FROM zws_middleware.zl_decision_info_log;
SELECT channel_code FROM zws_middleware.zl_decision_info_log;
SELECT distinct channel_code FROM zws_middleware.zl_decision_info_log;

SELECT * FROM zws_middleware.zl_decision_info_log;  -- 可查
SELECT channel_code FROM zws_middleware.zl_decision_info_log;
SELECT * FROM zws_middleware.channel_code_map;


select * from zx_credit_info;
select * from amt.product_credit_amt_detail;

select credit_amt,used_amt from zx_credit_info where customer_no = '';

-- 查询额度系统
select t.credit_amount from amt.product_credit_amt_detail t where t.product_code = 'ZL_APP' and t.customer_no = '';

-- 风控借款
SELECT * FROM zws_middleware.zl_drms_judgements;

SELECT * FROM zws_middleware.zx_credit_info;
SELECT customer_no FROM zws_middleware.zx_credit_info;

SELECT * FROM zws_middleware.zx_credit_applicant_result;
SELECT * FROM zws_middleware.zl_decision_info_log;
-- 13581666571
select * from amt_center.product_credit_amt_detail where customer_no='CT1849157812857798656';
select * from zws_middleware.zx_credit_user_info where mobile = '13581666571';
SELECT * FROM zws_middleware.zl_decision_info_log ORDER BY update_time DESC;


-- 有customer_no字段的中间表
SELECT
	channel_code AS traffic_channel_var,
    CASE
        WHEN channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC')
        THEN '众利（ZL）'
        WHEN channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT')
        THEN '天源花（TYH）'
    END AS user_source_var,
	final_deal_type_code AS result_var,
	refuse_msg AS reason_var,
	(select customer_no from zws_middleware.zx_credit_user_info where md5(mobile)=a.phone_md5 and customer_no in (select customer_no from zws_middleware.zx_credit_applicant_result)) as customer_no
	FROM zws_middleware.zl_decision_info_log a;




select a.customer_no from zws_middleware.zx_credit_user_info a join zws_middleware.zx_credit_applicant_result b on a.customer_no = b.customer_no;
select customer_no from zws_middleware.zx_credit_user_info where customer_no in (select customer_no from zws_middleware.zx_credit_applicant_result);
select customer_no from zws_middleware.zx_credit_user_info where customer_no in (select customer_no from zws_middleware.zx_credit_applicant_result) and md5(mobile)='';


-- 通过手机号获取出客户号
select customer_no from zx_credit_user_info where mobile = '';

-- 查询是否内部用户  > 0
select count(1) from zx_credit_applicant_result where customer_no = '';


select * from zws_middleware.zx_credit_user_info;
select * from zws_middleware.zx_credit_applicant_result;

select * from zws_middleware.zx_credit_user_info where md5(mobile)=''

-- 1.查询核心客户号
select credit_amt,used_amt from zx_credit_info where customer_no = '';
select * from zws_middleware.zx_credit_info;
select t.credit_amount from amt.product_credit_amt_detail t where t.product_code = 'ZL_APP' and t.customer_no = '';

-- 2.额度系统
-- 2.1先查询 客户维度 总额度-已用额度 = 可用额度1
-- total_credit_amount - sum(t.used_amount) = 可用额度1
select t.total_credit_amount from amt.customer_credit_amount t where  t.customer_no = '';
select sum(t.used_amount)  from amt.product_credit_amt_detail t where  t.customer_no = '';
-- total_credit_amount - sum(t.used_amount) = 可用额度1




select * from amt_center.customer_credit_amount;
select * from amt_center.product_credit_amt_detail;



-- 2.2后查询 产品维度 总额度-已用额度 = 可用额度2
-- credit_amount - used_amount =  可用额度2
select t.credit_amount,t.used_amount, from amt.product_credit_amt_detail t where t.product_code = 'ZL_APP' and t.customer_no = '';

select * from amt_center.product_credit_amt_detail;


-- 2.3最后
-- 真实可用额度 = 可用额度1 与 可用额度2 对比，取最小值


-- 主查询，结合第一个 SQL 的结果、授信额度和真实可用额度
SELECT
    -- 第一个 SQL 的结果
    a.traffic_channel_var,
    a.user_source_var,
    a.result_var,
    a.reason_var,
    a.customer_no,
    -- 授信额度
    COALESCE(credit_limit.credit_amount, 0) AS credit_limit,
    -- 真实可用额度
    LEAST(
        COALESCE(customer_credit.total_credit_amount, 0) - COALESCE(product_used.used_amount_sum, 0),
        COALESCE(credit_info.credit_amount, 0) - COALESCE(credit_info.used_amount, 0)
    ) AS real_available_limit
FROM (
    -- 第一个 SQL
    SELECT
        channel_code AS traffic_channel_var,
        CASE
            WHEN channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                                  'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC')
            THEN '众利（ZL）'
            WHEN channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
            THEN '龙力花（LLH）'
            WHEN channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                                  'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                                  'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT')
            THEN '天源花（TYH）'
        END AS user_source_var,
        final_deal_type_code AS result_var,
        refuse_msg AS reason_var,
        (
            SELECT customer_no
            FROM zws_middleware.zx_credit_user_info
            WHERE md5(mobile) = a.phone_md5
              AND customer_no IN (SELECT customer_no FROM zws_middleware.zx_credit_applicant_result)
        ) AS customer_no
    FROM zws_middleware.zl_decision_info_log a
) a
-- 左连接查询授信额度
LEFT JOIN (
    SELECT
        customer_no,
        credit_amount
    FROM
        amt_center.product_credit_amt_detail
    WHERE
        product_code = 'ZL_APP'
) credit_limit ON a.customer_no = credit_limit.customer_no
-- 左连接查询总授信额度
LEFT JOIN (
    SELECT
        customer_no,
        total_credit_amount
    FROM
        amt_center.customer_credit_amount
) customer_credit ON a.customer_no = customer_credit.customer_no
-- 左连接查询已使用额度总和
LEFT JOIN (
    SELECT
        customer_no,
        SUM(used_amount) AS used_amount_sum
    FROM
        amt_center.product_credit_amt_detail
    GROUP BY
        customer_no
) product_used ON a.customer_no = product_used.customer_no
-- 左连接获取 credit_amount 和 used_amount
LEFT JOIN (
    SELECT
        customer_no,
        credit_amount,
        used_amount
    FROM
        amt_center.product_credit_amt_detail
    WHERE
        product_code = 'ZL_APP'
) credit_info ON a.customer_no = credit_info.customer_no;




-- =======================
-- 15	creditRiskControlResult	风控授信结果返回

-- 风控授信
SELECT * FROM zl_drms_judgements WHERE apply_type= 'open_card' ;
select c.* FROM zx_credit_info c LEFT JOIN zx_credit_applicant_result r on c.applicant_result_id = r.id where r.customer_no = ''
-- 风控授信
SELECT * FROM zl_drms_judgements WHERE apply_type= 'open_card' ;

select c.* FROM zws_middleware.zx_credit_info c
LEFT JOIN zws_middleware.zx_credit_applicant_result r on c.applicant_result_id = r.id
LEFT JOIN zws_middleware.zl_drms_judgements j on j.apply_id = r.apply_id
where j.apply_type= 'open_card' and j.check_result = 'pass';


-- 回捞 情况 没有手机号
-- 自然授信 有手机号
--
 SELECT over_due_days FROM zws_middleware.zx_loan_note_info where status != 'FP' and  customer_no = 'CT1864935622192893952' and over_due_days > 0 ORDER BY over_due_days desc limit 1

select * from zws_middleware.zl_drms_judgements;
select * from zws_middleware.zl_drms_judgements where user_id is NULL ;
select * from zws_middleware.zx_credit_user_info where mobile is NULL ;
select * from zws_middleware.zx_credit_user_info;

SELECT * FROM zws_middleware.zx_credit_applicant_result;

SELECT
    *,
    check_result AS result_var,
    '' AS reason_var,
    channel_code AS trafficChannel_var,
    CASE
        WHEN channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC')
        THEN '众利（ZL）'
        WHEN channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT')
        THEN '天源花（TYH）'
    END AS user_source_var,
    '风控授信' AS creditType_var,
    EXTRACT(EPOCH FROM update_time) - EXTRACT(EPOCH FROM create_time) AS diff_in_seconds
FROM zws_middleware.zl_drms_judgements;



select
c.*,
c.credit_amt as creditLimit,
CAST(ROUND(CAST(c.credit_amt AS NUMERIC ) - CAST(c.used_amt AS NUMERIC),2) AS DECIMAL(10, 2)) as availableLimit
FROM zws_middleware.zx_credit_info c
LEFT JOIN zws_middleware.zx_credit_applicant_result r on c.applicant_result_id = r.id
LEFT JOIN zws_middleware.zl_drms_judgements j on j.apply_id = r.apply_id
where j.apply_type= 'open_card' and j.check_result = 'pass';





select
c.*,
c.credit_amt as creditLimit,
CAST(ROUND(CAST(c.credit_amt AS NUMERIC ) - CAST(c.used_amt AS NUMERIC),2) AS DECIMAL(10, 2)) as availableLimit
FROM zws_middleware.zx_credit_info c
LEFT JOIN zws_middleware.zx_credit_applicant_result r on c.applicant_result_id = r.id
LEFT JOIN zws_middleware.zl_drms_judgements j on j.apply_id = r.apply_id
where j.apply_type= 'open_card' and j.check_result = 'pass';



SELECT
    -- 第一条 SQL 中的字段选取和计算逻辑
    -- j.*,
    j.check_result AS result_var,
    '' AS reason_var,
    j.channel_code AS trafficChannel_var,
    CASE
        WHEN j.channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC')
        THEN '众利（ZL）'
        WHEN j.channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN j.channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT')
        THEN '天源花（TYH）'
    END AS user_source_var,
    '风控授信' AS creditType_var,
    EXTRACT(EPOCH FROM j.update_time) - EXTRACT(EPOCH FROM j.create_time) AS diff_in_seconds,
    -- 第二条 SQL 中的字段选取和计算逻辑
    c.credit_amt AS creditLimit,
    CAST(ROUND(CAST(c.credit_amt AS NUMERIC) - CAST(c.used_amt AS NUMERIC), 2) AS DECIMAL(10, 2)) AS availableLimit
FROM
    zws_middleware.zx_credit_info c
LEFT JOIN
    zws_middleware.zx_credit_applicant_result r
ON
    c.applicant_result_id = r.id
LEFT JOIN
    zws_middleware.zl_drms_judgements j
ON
    j.apply_id = r.apply_id
WHERE
    j.apply_type = 'open_card' AND j.check_result = 'pass';



select
	j.*,
	j.check_result as result_var,
	'' as reason_var,
	j.channel_code as trafficChannel_var,
    CASE
		WHEN j.channel_code = 'APPZY' THEN '众利（ZL）'
		WHEN j.channel_code = 'ICE_ZLSK_36' THEN '众利（ZL）'
		WHEN j.channel_code = 'ZL_HALUO' THEN '众利（ZL）'
		WHEN j.channel_code = 'HL_RL' THEN '众利（ZL）'
		WHEN j.channel_code = 'HY' THEN '众利（ZL）'
		WHEN j.channel_code = 'JD_RL' THEN '众利（ZL）'
		WHEN j.channel_code = 'QXL' THEN '众利（ZL）'
		WHEN j.channel_code = 'RS' THEN '众利（ZL）'
		WHEN j.channel_code = 'RS_RL' THEN '众利（ZL）'
		WHEN j.channel_code = 'VIVO_RL' THEN '众利（ZL）'
		WHEN j.channel_code = 'XL' THEN '众利（ZL）'
		WHEN j.channel_code = 'XL_LY' THEN '众利（ZL）'
		WHEN j.channel_code = 'YQG' THEN '众利（ZL）'
		WHEN j.channel_code = 'ZL_GM' THEN '众利（ZL）'
		WHEN j.channel_code = 'ZL_HSR' THEN '众利（ZL）'
		WHEN j.channel_code = 'ZL_HY' THEN '众利（ZL）'
		WHEN j.channel_code = 'ZL_WC' THEN '众利（ZL）'

		WHEN j.channel_code = 'LLH_HSR' THEN '龙力花（LLH）'
		WHEN j.channel_code = 'LLH_R360' THEN '龙力花（LLH）'
		WHEN j.channel_code = 'LLH_RP' THEN '龙力花（LLH）'
		WHEN j.channel_code = 'LLH_XY' THEN '龙力花（LLH）'

		WHEN j.channel_code = 'LXJ' THEN '天源花（TYH）'
		WHEN j.channel_code = 'R360' THEN '天源花（TYH）'
		WHEN j.channel_code = 'RP' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_APPZY' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_HSR' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_HY' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_JKQB' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_JQB' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_JXC' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_KN' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_LXJ' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_LXJPLUS' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_R360' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_RP' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_RPPLUS' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_SD' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_XD' THEN '天源花（TYH）'
		WHEN j.channel_code = 'TYH_XY' THEN '天源花（TYH）'
		WHEN j.channel_code = 'ZZJT' THEN '天源花（TYH）'
	END AS user_source_var,
	'风控授信' as creditType_var,
	EXTRACT(EPOCH FROM (j.update_time - j.create_time)) AS diff_in_seconds
    , r.id
    ,c.credit_amt as creditLimit,
CAST(ROUND(CAST(c.credit_amt AS NUMERIC ) - CAST(c.used_amt AS NUMERIC),2) AS DECIMAL(10, 2)) as availableLimit

 from zws_middleware.zl_drms_judgements  j
 LEFT JOIN zws_middleware.zx_credit_applicant_result r on j.apply_id = r.apply_id

 left join zws_middleware.zx_credit_info c on c.applicant_result_id = r.id
 ;



SELECT
    -- 第一条 SQL 中的字段选取和计算逻辑
    -- j.*,
    j.check_result AS result_var,
    '' AS reason_var,
    j.channel_code AS trafficChannel_var,
    CASE
        WHEN j.channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC')
        THEN '众利（ZL）'
        WHEN j.channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN j.channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT')
        THEN '天源花（TYH）'
    END AS user_source_var,
    '风控授信' AS creditType_var,
    EXTRACT(EPOCH FROM j.update_time) - EXTRACT(EPOCH FROM j.create_time) AS diff_in_seconds,
    -- 第二条 SQL 中的字段选取和计算逻辑
    c.credit_amt AS creditLimit,
    CAST(ROUND(CAST(c.credit_amt AS NUMERIC) - CAST(c.used_amt AS NUMERIC), 2) AS DECIMAL(10, 2)) AS availableLimit
FROM
    zws_middleware.zl_drms_judgements j
LEFT JOIN
    zws_middleware.zx_credit_applicant_result r
ON
    j.apply_id = r.apply_id
LEFT JOIN
    zws_middleware.zx_credit_info c
ON
    c.applicant_result_id = r.id
WHERE
    j.apply_type = 'open_card' AND j.check_result = 'pass';





-- =======================
-- 22	withdrawalRiskControlResult	风控借款审核结果返回
-- 风控借款 (简单)
SELECT * FROM zl_drms_judgements WHERE apply_type != 'open_card' ;

SELECT * FROM zws_middleware.zl_drms_judgements;



SELECT check_result as result_var,
'' as reason_var,
user_credit_amount_total as creditLimit,
user_credit_amount_available as availableLimit,
channel_code as trafficChannel_var,
    CASE
        WHEN channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC')
        THEN '众利（ZL）'
        WHEN channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT')
        THEN '天源花（TYH）'
    END AS user_source_var,
 FROM zws_middleware.zl_drms_judgements
 where apply_type <> 'open_card';


-- =======================
-- 用户属性

-- 通过手机号获取出客户号
select customer_no from zx_credit_user_info where mobile = '';
select customer_no from zws_middleware.zx_credit_user_info;

-- 在贷最大逾期天数
 SELECT over_due_days FROM zx_loan_note_info where status != 'FP' and  customer_no = 'CT1892096672664190976' and over_due_days > 0 ORDER BY over_due_days desc limit 1
SELECT over_due_days FROM zws_middleware.zx_loan_note_info where status != 'FP' and over_due_days > 0 ORDER BY over_due_days desc limit 1

-- 历史最大逾期天数
SELECT overdue_day from zx_loan_plan_info where overdue_day > 0 and  applicant_id in
(
 SELECT applicant_id FROM zx_loan_note_info where  customer_no = 'CT1892096672664190976'

)  ORDER BY overdue_day desc limit 1

SELECT overdue_day from zws_middleware.zx_loan_plan_info where overdue_day > 0 and  applicant_id in
(
 SELECT applicant_id FROM zws_middleware.zx_loan_note_info where  customer_no = 'CT1892096672664190976'

)  ORDER BY overdue_day desc limit 1





-- 设置hologres的binlog

--1
begin;
call set_table_property('zws_middleware.zx_repayment_apply_record', 'binlog.level', 'replica');
commit;

begin;
call set_table_property('zws_middleware.zx_repayment_apply_record', 'binlog.ttl', '259200');
commit;
--2
begin;
call set_table_property('zws_middleware.zl_decision_info_log', 'binlog.level', 'replica');
commit;

begin;
call set_table_property('zws_middleware.zl_decision_info_log', 'binlog.ttl', '259200');
commit;

select * from zws_middleware.zl_decision_info_log;



select * from zl_data_clean.prod_risk_idno_dtr;


SELECT * FROM zl_data_clean.prod_risk_idno_dtr;
select * from zws_middleware.zx_credit_user_info;


-- =======================
zl_data_clean.prod_risk_idno_dtr;
这个与 userCollisionRequestResult	用户撞库结果

select * from zl_data_clean.prod_risk_idno_dtr;



-- 用户属性 - 旧
-- data_realtime 这个库
select * from realtime_dwd.dwd_user_info_real;  -- channel_key 当channel_code

-- select * from realtime.dwd_user_info_real;

select * from zws_middleware.zl_area_code;
select * from zws_middleware.zx_credit_info;
select * from zws_middleware.zx_loan_plan_info;
select * from zws_middleware.zx_loan_apply_record;

-- user_info_view,credit_amt_view,max_overdue_view
select * from realtime.gio_event_message;


select * from zl_fk_data.temp_channel_product_mapping2;

select * from zws_middleware.zx_credit_user_info;


select * from amt_center.product_credit_amt_detail;



select * from realtime_dwd.dwd_user_info_real;
select count(*) from realtime_dwd.dwd_user_info_real GROUP BY cust_no;


-- =======================
-- 用户属性
-- 用户信息表
select * from realtime_dwd.dwd_user_info_real;

select * from zws_middleware.zl_area_code;

-- 360智信授信申请_返回结果的授信信息
select * from zws_middleware.zx_credit_info;

--
select * from zws_middleware.zx_loan_plan_info;

-- 众利-风控决策-提交申请
select * from zws_middleware.zl_drms_judgements;

-- 借款申请记录表
select * from zws_middleware.zx_loan_apply_record;



select * from realtime.gio_event_message ORDER BY event_time DESC;


select * from realtime.gio_event_message where user_source_var !='众利（ZL）' ORDER BY event_time DESC;
select * from realtime.gio_event_message where user_source_var ='众利（ZL）' ORDER BY event_time DESC;


select * from zws_middleware.zx_credit_user_info;

select channel_code, count(*) from zws_middleware.zx_credit_user_info  where mobile='13076844083' GROUP BY channel_code;

select channel_code, GROUP_CONCAT(CASE
        WHEN z.channel_code IN (
            'APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
            'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC'
        )
        THEN '众利（ZL）'
        WHEN z.channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN z.channel_code IN (
            'LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
            'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
            'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT'
        )
        THEN '天源花（TYH）'
        ELSE NULL
    END) AS business_line  from zws_middleware.zx_credit_user_info z  where mobile='13076844083' GROUP BY channel_code;

select mobile, count(*) from zws_middleware.zx_credit_user_info GROUP BY mobile;

select * from zws_middleware.zx_credit_user_info where mobile='13076844083';

select * from zws_middleware.realtime_dwd.dwd_user_info_real where mobile='13076844083'


select * from zws_middleware.zx_credit_user_info where mobile='13076844083';


select channel_code
, STRING_AGG(CASE
        WHEN z.channel_code IN (
            'APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
            'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC'
        )
        THEN '众利（ZL）'
        WHEN z.channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN z.channel_code IN (
            'LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
            'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
            'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT'
        )
        THEN '天源花（TYH）'
        ELSE NULL
    END , ', ') AS business_line  from zws_middleware.zx_credit_user_info z  where mobile='13076844083' GROUP BY z.channel_code;



select mobile
, STRING_AGG(DISTINCT CASE
        WHEN z.channel_code IN (
            'APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
            'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC'
        )
        THEN '众利（ZL）'
        WHEN z.channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY')
        THEN '龙力花（LLH）'
        WHEN z.channel_code IN (
            'LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
            'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
            'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT'
        )
        THEN '天源花（TYH）'
        ELSE NULL
    END , ', ') AS business_line  from zws_middleware.zx_credit_user_info z  where mobile='18707255217' GROUP BY z.mobile;


select * from zws_middleware.zl_decision_info_log order by update_time DESC;


select count(*) from realtime.gio_user_message;
select * from realtime.gio_user_message order by event_time DESC;;
select * from realtime.gio_user_message where login_user_id='13233806000';

select * from zws_middleware.zx_credit_user_info where mobile='13825588246'
select * from realtime.gio_user_message where login_user_id='13993607539';
select * from realtime.gio_user_message where available_limit_ppl is not NULL  order by event_time DESC;;


select * from mongo_ods.fl_duyi_score;

select * from mongo_ods.fl_duyi_score_req;

select * from zws_middleware.zx_credit_info where ;
select * from zws_middleware.zl_drms_judgements where user_mobile_number='18510271504';
select * from zws_middleware.zx_credit_user_info where mobile='18510271504';


select * from zws_middleware.zl_drms_judgements;


select * from zws_middleware.zl_drms_judgements  where user_id ='RS_RLU1900097850092826624'
select * from zws_middleware.zl_decision_info_log where channel_code='RS_RL' order by create_time desc limit 10


select * from realtime_dwd.dwd_user_info_real where mobile_no='18707255217';
select * from zws_middleware.zx_credit_user_info where mobile='18707255217';
select mobile,count(*) from zws_middleware.zx_credit_user_info  group by mobile;

select * from customer_center.customer_base_info where customer_no='CT1875042234677428224'
select * from customer_center.c_base_info_all where cust_no='CT1875042234677428224'
select * from customer_center.c_base_info_2;

select TO_TIMESTAMP(hg_binlog_timestamp_us/1000000) from zws_middleware.zx_credit_user_info where mobile='18707255217';
select hg_binlog_timestamp_us from zws_middleware.zx_credit_user_info where mobile='18707255217';
1741854772149213
select * from realtime.gio_user_message where login_user_id='18707255217';

select hg_binlog_timestamp_us from realtime_dwd.dwd_user_info_real where mobile_no='18707255217';

1741713776873699
1741713776873699
1741854772149213

1741854772722347
1741854777425827
1741854777425827
1741710234102104
1741710234102104
1741717478949211
1741717478949211

select * from realtime.gio_user_message where history_overdue_days_ppl is not NULL ;
select * from realtime.gio_user_message;
 SELECT over_due_days FROM zws_middleware.zx_loan_note_info where status != 'FP' and  customer_no = 'CT1862318645800878080' and over_due_days > 0 ORDER BY over_due_days desc ;




CREATE TABLE zl_fk_data.temp_channel_product_mapping2 (
    channel_code text,
    product_code text,
    source_code text,
    PRIMARY key(channel_code)
)with (
orientation = 'column',
storage_format = 'orc',
bitmap_columns = 'channel_code,product_code,source_code'
);

SELECT * FROM zl_fk_data.temp_channel_product_mapping2;

INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('HY', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_LXJPLUS', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('R360', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('XL', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('ICE_ZLSK_36', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_KN', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('LLH_XY', 'LLH_API', '龙力花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_XD', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('JD_RL', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_R360', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('HL_RL', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_LXJ', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_HY', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('ZL_WC', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('RS_RL', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('ZL_GM', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_JQB', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('APPZY', 'ZL_APP', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_JXC', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('HLCX', 'HALUO', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_HSR', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('ZL_HY', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_APPZY', 'TYH_APP', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('ZL_HSR', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('RP', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('XL_LY', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('RS', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('YQG', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_JKQB', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('ZZJT', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('LLH_R360', 'LLH_API', '龙力花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_RPPLUS', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('QXL', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('VIVO_RL', 'ZL_API', '众利');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('LLH_HSR', 'LLH_API', '龙力花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_RP', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('LLH_RP', 'LLH_API', '龙力花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_SD', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('LXJ', 'TYH_API', '天源花');
INSERT INTO zl_fk_data.temp_channel_product_mapping2(channel_code, product_code, source_code) VALUES ('TYH_XY', 'TYH_API', '天源花');


select * from zl_fk_data.temp_channel_product_mapping2;

-- 是否授信,如果有记录表明授信过,按照channel_code 分组
select channel_code, count(1) from zws_middleware.zx_credit_user_info where mobile ='13923141167'  GROUP BY channel_code;

select channel_code, count(1) from zws_middleware.zx_credit_user_info where mobile ='13923141167'  GROUP BY channel_code;
select channel_code, count(1) from zws_middleware.zx_credit_user_info where mobile ='13836518334'  GROUP BY channel_code;
select * from zws_middleware.zx_credit_user_info where mobile ='13836518334';
select * from zws_middleware.zx_credit_user_info;
select * from zws_middleware.zx_credit_info;

select * from realtime_dwd.dwd_user_info_real;

-- 阿里云Flink采用的是基于Apache Flink增强的企业级引擎Ververica Runtime（简称VVR）。
-- VVR使用三位编号的方案来指定阿里云Flink产品引擎版本的发布版本。引擎版本的格式为Major1.Major2.Minor。版本号中的Major和Minor含义详情如下：
-- Major部分：表示VVR功能的根本变化和增加。我们会根据每个引擎版本中更改的大小和规模，而增加对应Major1和Major2部分的数字。
-- Minor部分：表示质量改进和对现有功能的修复。当许多质量改进被添加到版本中时，我们会递增Minor部分的数字。

select * from amt_center.product_credit_amt_detail;
-- realtime.dwd_user_info_real 是通过zws_middleware.zx_credit_id_card_ocr_info 和 zws_middleware.zx_credit_user_info 两个表合并的，第二个表会覆盖第一个表的相同字段。

-- amt_center.product_credit_amt_detail 额度中心，额度中心amt_center有多个表，额度中心和核心什么关系
-- amt_center.product_credit_amt_detail 产品授信额度明细表
-- amt_center.channel_info_config 渠道信息配置表
-- product_credit_amt_detail_source 直接amt_center.product_credit_amt_detail，



CREATE TEMPORARY VIEW __viewName_1
AS SELECT t1.*
    ,t2.credit_amount   -- APP产品授信额度
    ,t2.used_amount   -- App产品已用额度
    ,(t2.credit_amount - t2.used_amount) as available_limit -- 产品域可用额度

FROM decision_user_view as  t1

LEFT JOIN product_credit_amt_detail_source FOR SYSTEM_TIME AS OF PROCTIME() as t2
 ON t1.customer_no = t2.customer_no and t2.product_code = t1.product_code
WHERE t1.mobile is not NULL
;

amt_center.product_credit_amt_detail

select hg_binlog_timestamp_us from realtime_dwd.dwd_user_info_real where mobile_no='18707255217';


select * from amt_center.product_credit_amt_detail;
select customer_no,count(*) from amt_center.product_credit_amt_detail group by customer_no;
select * from amt_center.product_credit_amt_detail where customer_no='CT1892032354118512640';
select * from amt_center.product_credit_amt_detail where customer_no='CT1826183752922324992';




select * from zws_middleware.zx_credit_applicant_result; -- id
select * from zws_middleware.zx_credit_user_info; -- applicant_id




select t1.*,t2.product_code,t2.source_code，t3.credit_amount,t3.used_amount
from zws_middleware.zx_credit_user_info t1
left join zl_fk_data.temp_channel_product_mapping2 t2 on t1.channel_code=t2.channel_code
left join amt_center.product_credit_amt_detail t3 on t1.customer_no=t3.customer_no;

SELECT
    t1.*,
    t2.product_code,
    t2.source_code,  -- 如果 source_code 是一个字段，请直接引用它
    t3.credit_amount,
    t3.used_amount
FROM
    zws_middleware.zx_credit_user_info t1
LEFT JOIN
    zl_fk_data.temp_channel_product_mapping2 t2
    ON t1.channel_code = t2.channel_code
LEFT JOIN
    amt_center.product_credit_amt_detail t3
    ON t1.customer_no = t3.customer_no;

select * from zws_middleware.zx_credit_user_info;
select * from zl_fk_data.temp_channel_product_mapping2;
select * from amt_center.product_credit_amt_detail;


select * from mongo_ods.tengrui_r904;



-- businessLine	业务线
select product_line from zws_middleware.zl_api_user where channel_code = 'HY';
select * from zws_middleware.zl_api_user;

-- creditLimit_ppl	app域用户授信额度
-- 2.额度系统
-- 2.1先查询 客户维度 总额度-已用额度 = 可用额度1
-- total_credit_amount - sum(t.used_amount) = 可用额度1
SELECT
    t.total_credit_amount
FROM
    amt.customer_credit_amount t
WHERE
    t.customer_no = '';

SELECT
    sum(t.used_amount)
FROM
    amt.product_credit_amt_detail t
WHERE
    t.customer_no = '';

-- 2.2后查询 产品维度 总额度-已用额度 = 可用额度2
-- credit_amount - used_amount =  可用额度2

SELECT
    t.credit_amount,
    t.used_amount,
FROM
    amt.product_credit_amt_detail t
WHERE
    t.product_code = 'ZL_APP'
    AND t.customer_no = '';


-- 2.3最后
-- 真实可用额度 = 可用额度1 与 可用额度2 对比，取最小值


-- 查询额度系统； availableLimit_ppl	app域用户可用额度
SELECT
    t.credit_amount
FROM
    amt.product_credit_amt_detail t
WHERE
    t.product_code = 'ZL_APP'
    AND t.customer_no = '';



-- crediStatus_ppl	app域额度状态
SELECT
    `status`
FROM
    amt_center.product_credit_amt_detail
WHERE
    customer_no = 'CT1891306592433377280'
    AND product_code = 'ZL_APP'
    AND amount_type = '1';

select * from realtime_dwd.dwd_user_info_real;
select count(*) from realtime_dwd.dwd_user_info_real;  -- 13539278


SELECT count(*) FROM amt_center.product_credit_amt_detail; -- 2362751

SELECT * FROM amt_center.product_credit_amt_detail;
SELECT * FROM amt_center.product_credit_amt_detail where product_code='ZL_APP';

SELECT status FROM amt_center.product_credit_amt_detail GROUP BY status;

select * from  amt_center.product_credit_amt_detail where customer_no='CT1825883338008051712';


-- ifCredit_ppl	最近一次授信是否成功
-- status S 授信成功 F 失败 P 处理中
SELECT  `status`
FROM    zws_middleware.zx_credit_applicant_result
WHERE   customer_no = 'CT1867044792292646912'
AND     channel_code IN (
            SELECT  channel_code
            FROM    zws_middleware.zl_api_user
            WHERE   product_line IN (
                        SELECT  product_line
                        FROM    zl_api_user
                        WHERE   channel_code = 'TYH_HY'
                    )
        )
ORDER BY id DESC
LIMIT   1
;

select * from zws_middleware.zl_api_user;


select * from zws_middleware.zx_credit_user_info;
select * from zws_middleware.zx_credit_applicant_result;


select status from zws_middleware.zx_credit_applicant_result GROUP BY status;


select * from amt_center.customer_credit_amount;

select * from amt_center.product_credit_amt_detail;


-- 问题
-- 1，credit_user_info 到 zx_credit_applicant_result ，时间间隔，zx_credit_applicant_result触发会多次上报，不需要channel_code IN吧，这里的逻辑是？
-- 2，zl_api_user这里








-- 查询额度系统 授信额度
SELECT
    t.credit_amount
FROM
    amt.product_credit_amt_detail t
WHERE
    t.product_code = 'ZL_APP'
    AND t.customer_no = '';

-- 可用额度
-- # 2.额度系统

-- # 2.1先查询 客户维度 总额度-已用额度 = 可用额度1

-- # total_credit_amount - sum(t.used_amount) = 可用额度1

select t.total_credit_amount from amt.customer_credit_amount t where  t.customer_no = '';
select sum(t.used_amount)  from amt.product_credit_amt_detail t where  t.customer_no = '';

-- # 2.2后查询 产品维度 总额度-已用额度 = 可用额度2

-- # credit_amount - used_amount =  可用额度2
select t.credit_amount,t.used_amount, from amt.product_credit_amt_detail t where t.product_code = 'ZL_APP' and t.customer_no = '';



-- # 2.3最后

-- 真实可用额度 = 可用额度1 与 可用额度2 对比，取最小值


-- ===================================

-- 查询额度系统 - creditLimit_ppl	app域用户授信额度
select t.credit_amount from amt.product_credit_amt_detail t where t.product_code = 'ZL_APP' and t.customer_no = '';
select t.credit_amount,t.used_amount,(t.credit_amount-t.used_amount) as available_amt from amt_center.product_credit_amt_detail t where t.product_code = 'ZL_APP' and t.customer_no = '';

select t.credit_amount,t.used_amount,(t.credit_amount-t.used_amount) as available_amt from amt_center.product_credit_amt_detail t where t.product_code = 'ZL_APP';

-- crediStatus_ppl	app域额度状态
SELECT
    `status`
FROM
    product_credit_amt_detail
WHERE
    customer_no = 'CT1891306592433377280'
    AND product_code = 'ZL_APP'
    AND amount_type = '1';


-- ifCredit_ppl	最近一次授信是否成功
-- ## status S 授信成功 F 失败 P 处理中
-- ## status S 授信成功 F 失败 P 处理中 R:授信拒绝 U:该申请号不存在（掉单）
SELECT
    `status`
FROM
    zx_credit_applicant_result
WHERE
    customer_no = 'CT1867044792292646912'
    AND status != 'P'
    AND channel_code IN (
        SELECT
            channel_code
        FROM
            zl_api_user
        WHERE
            product_line IN (
                SELECT
                    product_line
                FROM
                    zl_api_user
                WHERE
                    channel_code = 'TYH_HY'))
    ORDER BY
        id DESC
    LIMIT 1;

SELECT
    `status`
FROM
    product_credit_amt_detail
WHERE
    customer_no = 'CT1891306592433377280'
    AND status != 'P'
    AND product_code = 'ZL_APP'
    AND amount_type = '1';

select customer_no from zws_middleware.zx_credit_user_info;
select * from zws_middleware.zx_credit_applicant_result;
SELECT
    *
FROM
    zws_middleware.zx_credit_user_info t1
    LEFT JOIN zws_middleware.zx_credit_applicant_result t2
	on t1.customer_no=t2.customer_no
	ORDER BY t2.id DESC;

select t1.*, case WHEN (select status from zws_middleware.zx_credit_applicant_result t2 where t2.customer_no=t1.customer_no and t1.status != 'P' order by id desc limit 1) as status  from zws_middleware.zx_credit_user_info t1;


-- 业务线 businessLine	业务线
select product_line from zl_api_user where channel_code = 'HY';
select * from zws_middleware.zl_api_user;
-- 42
select count(*) from zws_middleware.zl_api_user;



-- 用户撞库
select * from zws_middleware.zl_decision_info_log order by create_time DESC;
select * from zws_middleware.zx_credit_user_info order by create_time DESC;

select * from realtime.gio_event_message where event_key='userCollisionRequestResult';
select * from realtime.gio_event_message where event_key='userCollisionRequestResult' ORDER BY event_time DESC;

select * from realtime.gio_event_message where event_key='creditRiskControlResult';
select * from realtime.gio_event_message where event_key='creditRiskControlResult' ORDER BY event_time DESC;

select * from realtime.gio_event_message where event_key='withdrawalRiskControlResult';
select * from realtime.gio_event_message where event_key='withdrawalRiskControlResult' ORDER BY event_time DESC;

select * from zl_fk_data.temp_channel_product_mapping2;


select * from zws_middleware.zl_drms_judgements;

select * from zws_middleware.zl_drms_judgements where user_mobile_number is NULL;


select * from realtime.gio_user_message_prd;
select * from realtime.gio_event_message_prd;

select * from atreus_es.atreus_entrysearch;
select * from atreus_es.atreus_entrysearch order by recordtime DESC limit 10;
select * from atreus_es.atreus_entrysearch where recordtime >=1743177600144;
select count(*) from atreus_es.atreus_entrysearch where recordtime >=1743177600144;  -- 1476307

--  489240 4月6日一整天，
select count(*) from atreus_es.atreus_entrysearch where recordtime <=1743955200000 and recordtime>=1743868800000 ;
-- 726507 4月7日一整天，
select count(*) from atreus_es.atreus_entrysearch where recordtime <=1744041600000 and recordtime>=1743955200000 ;



select * from atreus_es.atreus_entrysearch where recordtime >=1743177600144 ORDER BY recordtime DESC ; --1743523199481 2025-04-01 23:59:59

select * from atreus_es.atreus_entrysearch where recordtime <=1743177600144 ORDER BY recordtime DESC ; 1740585600618,1743177582071
2025-03-28 23:59:42
2025-02-27 00:00:00
2025-03-29 00:00:00
select * from atreus_es.atreus_entrysearch_temp;
select count(*) from atreus_es.atreus_entrysearch_temp;
-- DELETE FROM atreus_es.atreus_entrysearch_temp;
select count(*) from atreus_es.atreus_entrysearch;
delete from atreus_es.atreus_entrysearch;


-- 2025-04-02
select * from atreus_es.atreus_entrysearch_100;
-- 2025-04-02
select * from atreus_es.freyr_100;

--2025-03-29
select * from atreus_es.atreus_componentlog_100;
--2025-03-29
select * from atreus_es.atreus_entryinvokeresult_100;




-- TRUNCATE table atreus_es.freyr;
-- delete from atreus_es.freyr_100;
-- delete from atreus_es.atreus_entrysearch;




532927199104140514
select  t1.id_card_no from zl_fk_data.t_cust_black_list_update t1 join( select max(dt) as dt from  zl_fk_data.t_cust_black_list_update ) t2 on t1.dt=t2.dt;
-- 637611

select count(*) from (select  t1.id_card_no from zl_fk_data.t_cust_black_list_update t1 join( select max(dt) as dt from  zl_fk_data.t_cust_black_list_update ) t2 on t1.dt=t2.dt) as t;

select count(*) from (select  t1.id_card_no from zl_fk_data.t_cust_black_list_update t1 join( select max(dt) as dt from  zl_fk_data.t_cust_black_list_update ) t2 on t1.dt=t2.dt) as t where t.id_card_no='532927199104140514';

select count(*) from (select  t1.id_card_no from zl_fk_data.t_cust_black_list_update t1 join( select max(dt) as dt from  zl_fk_data.t_cust_black_list_update ) t2 on t1.dt=t2.dt) as t where t.id_card_no='532927199104140513';

select * from (select  t1.id_card_no from zl_fk_data.t_cust_black_list_update t1 join( select max(dt) as dt from  zl_fk_data.t_cust_black_list_update ) t2 on t1.dt=t2.dt) as t where t.id_card_no='532927199104140513';
select * from (select  t1.id_card_no from zl_fk_data.t_cust_black_list_update t1 join( select max(dt) as dt from  zl_fk_data.t_cust_black_list_update ) t2 on t1.dt=t2.dt) as t where t.id_card_no='532927199104140514';


select * from zl_data_clean.ads_anti_fraud_msg_record_var where mobile='17768912515';


select * from mongo_ods.baiwei_yhf_v31 limit 10;

BEGIN;

/*
DROP TABLE atreus_es.atreus_entrysearch;
*/

-- Type: TABLE ; Name: atreus_entrysearch; Owner: p4_201351809199932519

CREATE TABLE atreus_es.atreus_entrysearch (
    _id text NOT NULL,
    calldate bigint,
    strparams json,
    productname text,
    entryid text,
    finaldealtype text,
    entrystatus text,
    invokemode integer,
    workflowtype text,
    policyset json,
    statdate bigint,
    cost integer,
    test integer,
    productid text,
    customparams json,
    appname text,
    workflowname text,
    entryparams json,
    entryoccurtime bigint,
    token text NOT NULL,
    workflowcode text,
    addtype text,
    productcode text,
    recordtime bigint,
    finaldealtypename text,
    longparams json,
    entryparamsser text,
    workflowversion integer,
    thirdservice json,
    policysetscore json,
    workflowid text,
    wholeid text
    ,PRIMARY KEY (_id)
)with (
orientation = 'column',
storage_format = 'orc',
bitmap_columns = '_id,productname,entryid,finaldealtype,entrystatus,workflowtype,productid,appname,workflowname,token,workflowcode,addtype,productcode,finaldealtypename,entryparamsser,workflowid,wholeid',
dictionary_encoding_columns = '_id,productname:auto,entryid:auto,finaldealtype:auto,entrystatus:auto,workflowtype:auto,productid:auto,appname:auto,workflowname:auto,token:auto,workflowcode:auto,addtype:auto,productcode:auto,finaldealtypename:auto,entryparamsser:auto,workflowid:auto,wholeid:auto',
distribution_key = '_id',
table_group = 'data_dws_tg_default',
table_storage_mode = 'hot',
time_to_live_in_seconds = '3153600000'
);



COMMENT ON TABLE atreus_es.atreus_entrysearch IS NULL;
ALTER TABLE atreus_es.atreus_entrysearch OWNER TO p4_201351809199932519;


END;



SELECT
    DATE_TRUNC('hour', TO_TIMESTAMP(recordtime / 1000)) AS hour_start, -- 按小时截断时间
    COUNT(*) AS record_count -- 统计每小时的记录数
FROM
    atreus_es.atreus_entrysearch
WHERE
    recordtime>1743955200000 and recordtime<1744041600000
GROUP BY
    DATE_TRUNC('hour', TO_TIMESTAMP(recordtime / 1000)) -- 按小时分组
ORDER BY
    hour_start; -- 按时间排序




select * from realtime.risk_idno_channel_dtr_real where id_no = '' and channel_code='';

select distinct risk_channel_code,
case when risk_channel_code='TYH_APPZY_TYH_APPZY' then 'TYH_APPZY'
when risk_channel_code='TYH_RP' then 'TYH_RP' else channel_code end as channel_code
from zl_data_clean.dim_zl_channel_info
where risk_channel_code is not null;


select distinct risk_channel_code,
case when risk_channel_code='TYH_APPZY_TYH_APPZY' then 'TYH_APPZY'
when risk_channel_code='TYH_RP' then 'TYH_RP' else channel_code end as channel_code
from zl_data_clean.dim_zl_channel_info
where risk_channel_code = 'ZL_HSR';

select distinct risk_channel_code,
case when risk_channel_code='TYH_APPZY_TYH_APPZY' then 'TYH_APPZY'
when risk_channel_code='TYH_RP' then 'TYH_RP' else channel_code end as channel_code
from zl_data_clean.dim_zl_channel_info
where risk_channel_code is not null;


SELECT * FROM zl_data_clean.dim_zl_channel_info;
select * from realtime.risk_idno_channel_dtr_real;
select * from realtime.risk_idno_channel_dtr_real where id_no = '152128199612270616' and channel_code='TYH';

select * from atreus_es.atreus_componentlog_100;
select * from atreus_es.atreus_entryinvokeresult_100;
select * from atreus_es.freyr_100;



CREATE TABLE atreus_es.time_partitioned_table (
  id text,              -- 唯一标识符
  ms_time bigint,       -- 毫秒级时间戳
  other_data text       -- 其他字段
)
PARTITION BY RANGE (ms_time);


-- 子表 a: 2025年1月1日 - 2025年1月31日
CREATE TABLE atreus_es.partition_a PARTITION OF atreus_es.time_partitioned_table
FOR VALUES FROM (1735689600000) TO (1738367999999);

-- 子表 b: 2025年2月1日 - 2025年2月28日
CREATE TABLE atreus_es.partition_b PARTITION OF atreus_es.time_partitioned_table
FOR VALUES FROM (1738454400000) TO (1741046399999);

BEGIN;

-- Step 1: 创建父表
CREATE TABLE atreus_es.atreus_entrysearch_partitioned (
    _id text NOT NULL,
    calldate bigint,
    strparams json,
    productname text,
    entryid text,
    finaldealtype text,
    entrystatus text,
    invokemode integer,
    workflowtype text,
    policyset json,
    statdate bigint,
    cost integer,
    test integer,
    productid text,
    customparams json,
    appname text,
    workflowname text,
    entryparams json,
    entryoccurtime bigint,
    token text NOT NULL,
    workflowcode text,
    addtype text,
    productcode text,
    recordtime bigint, -- 毫秒时间戳
    finaldealtypename text,
    longparams json,
    entryparamsser text,
    workflowversion integer,
    thirdservice json,
    policysetscore json,
    workflowid text,
    wholeid text,
    partition_key text -- 新增的分区键字段
    ,PRIMARY KEY (_id, partition_key) -- 主键包含 partition_key
)
PARTITION BY LIST (partition_key) -- 使用 LIST 分区
WITH (
    orientation = 'column',
    storage_format = 'orc',
    bitmap_columns = '_id,productname,entryid,finaldealtype,entrystatus,workflowtype,productid,appname,workflowname,token,workflowcode,addtype,productcode,finaldealtypename,entryparamsser,workflowid,wholeid',
    dictionary_encoding_columns = '_id,productname:auto,entryid:auto,finaldealtype:auto,entrystatus:auto,workflowtype:auto,productid:auto,appname:auto,workflowname:auto,token:auto,workflowcode:auto,addtype:auto,productcode:auto,finaldealtypename:auto,entryparamsser:auto,workflowid:auto,wholeid:auto',
    distribution_key = '_id',
    table_group = 'data_dws_tg_default',
    table_storage_mode = 'hot',
    time_to_live_in_seconds = '3153600000'
);

COMMENT ON TABLE atreus_es.atreus_entrysearch_partitioned IS NULL;
ALTER TABLE atreus_es.atreus_entrysearch_partitioned OWNER TO p4_201351809199932519;

-- Step 2: 创建子表（示例：2025 年 1 月 和 2025 年 2 月）
CREATE TABLE atreus_es.atreus_entrysearch_202501 PARTITION OF atreus_es.atreus_entrysearch_partitioned
FOR VALUES IN ('202501');

CREATE TABLE atreus_es.atreus_entrysearch_202502 PARTITION OF atreus_es.atreus_entrysearch_partitioned
FOR VALUES IN ('202502');

END;


-- 插入数据
INSERT INTO atreus_es.atreus_entrysearch_partitioned (
    _id, calldate, strparams, productname, entryid, finaldealtype, entrystatus, invokemode, workflowtype, policyset, statdate, cost, test, productid, customparams, appname, workflowname, entryparams, entryoccurtime, token, workflowcode, addtype, productcode, recordtime, finaldealtypename, longparams, entryparamsser, workflowversion, thirdservice, policysetscore, workflowid, wholeid, partition_key
)
VALUES (
    'id_1',
    1735689600000,
    '{}',
    'ProductA',
    'entry1',
    'DealTypeA',
    'StatusA',
    1,
    'WorkflowTypeA',
    '{}',
    1735689600000,
    100,
    0,
    'product1',
    '{}',
    'AppNameA',
    'WorkflowNameA',
    '{}',
    1735689600000,
    'token1',
    'WorkflowCodeA',
    'AddTypeA',
    'ProductCodeA',
    1735689600000,
    'FinalDealTypeNameA',
    '{}',
    'EntryParamsSerA',
    1,
    '{}',
    '{}',
    'WorkflowIdA',
    'WholeIdA',
    '202501' -- 直接赋值 partition_key
);

select * from atreus_es.atreus_entrysearch_partitioned;
select * from atreus_es.atreus_entrysearch_202501;


BEGIN;

-- Step 1: 创建父表
CREATE TABLE atreus_es.atreus_entrysearch_partd (
    _id text NOT NULL,
    calldate bigint,
    strparams json,
    productname text,
    entryid text,
    finaldealtype text,
    entrystatus text,
    invokemode integer,
    workflowtype text,
    policyset json,
    statdate bigint,
    cost integer,
    test integer,
    productid text,
    customparams json,
    appname text,
    workflowname text,
    entryparams json,
    entryoccurtime bigint,
    token text NOT NULL,
    workflowcode text,
    addtype text,
    productcode text,
    recordtime bigint, -- 毫秒时间戳
    finaldealtypename text,
    longparams json,
    entryparamsser text,
    workflowversion integer,
    thirdservice json,
    policysetscore json,
    workflowid text,
    wholeid text,
    partd_key text -- 新增的分区键字段
    ,PRIMARY KEY (_id, partd_key) -- 主键包含 partd_key
)
PARTITION BY LIST (partd_key) -- 使用 LIST 分区
WITH (
    orientation = 'column',
    storage_format = 'orc',
    bitmap_columns = '_id,productname,entryid,finaldealtype,entrystatus,workflowtype,productid,appname,workflowname,token,workflowcode,addtype,productcode,finaldealtypename,entryparamsser,workflowid,wholeid',
    dictionary_encoding_columns = '_id,productname:auto,entryid:auto,finaldealtype:auto,entrystatus:auto,workflowtype:auto,productid:auto,appname:auto,workflowname:auto,token:auto,workflowcode:auto,addtype:auto,productcode:auto,finaldealtypename:auto,entryparamsser:auto,workflowid:auto,wholeid:auto',
    distribution_key = '_id',
    table_group = 'data_dws_tg_default',
    table_storage_mode = 'hot',
    time_to_live_in_seconds = '3153600000'
);

COMMENT ON TABLE atreus_es.atreus_entrysearch_partd IS NULL;
ALTER TABLE atreus_es.atreus_entrysearch_partd OWNER TO p4_201351809199932519;

-- Step 2: 创建子表（示例：2025 年 1 月 和 2025 年 2 月）
CREATE TABLE atreus_es.atreus_entrysearch_202501 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202501');

CREATE TABLE atreus_es.atreus_entrysearch_202502 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202502');

END;
CREATE TABLE atreus_es.atreus_entrysearch_202503 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202503');
CREATE TABLE atreus_es.atreus_entrysearch_202504 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202504');

CREATE TABLE atreus_es.atreus_entrysearch_202505 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202505');
CREATE TABLE atreus_es.atreus_entrysearch_202506 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202506');
CREATE TABLE atreus_es.atreus_entrysearch_202507 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202507');
CREATE TABLE atreus_es.atreus_entrysearch_202508 PARTITION OF atreus_es.atreus_entrysearch_partd
FOR VALUES IN ('202508');

-- 插入数据
INSERT INTO atreus_es.atreus_entrysearch_partd (
    _id, calldate, strparams, productname, entryid, finaldealtype, entrystatus, invokemode, workflowtype, policyset, statdate, cost, test, productid, customparams, appname, workflowname, entryparams, entryoccurtime, token, workflowcode, addtype, productcode, recordtime, finaldealtypename, longparams, entryparamsser, workflowversion, thirdservice, policysetscore, workflowid, wholeid, partd_key
)
VALUES (
    'id_1',
    1735689600000,
    '{}',
    'ProductA',
    'entry1',
    'DealTypeA',
    'StatusA',
    1,
    'WorkflowTypeA',
    '{}',
    1735689600000,
    100,
    0,
    'product1',
    '{}',
    'AppNameA',
    'WorkflowNameA',
    '{}',
    1735689600000,
    'token1',
    'WorkflowCodeA',
    'AddTypeA',
    'ProductCodeA',
    1735689600000,
    'FinalDealTypeNameA',
    '{}',
    'EntryParamsSerA',
    1,
    '{}',
    '{}',
    'WorkflowIdA',
    'WholeIdA',
    '202501' -- 直接赋值 partition_key
);

select * from atreus_es.atreus_entrysearch_partd;
select * from atreus_es.atreus_entrysearch_202501;

insert into atreus_es.atreus_entrysearch_partd(
    _id, calldate, strparams, productname, entryid, finaldealtype, entrystatus, invokemode, workflowtype, policyset, statdate, cost, test, productid, customparams, appname, workflowname, entryparams, entryoccurtime, token, workflowcode, addtype, productcode, recordtime, finaldealtypename, longparams, entryparamsser, workflowversion, thirdservice, policysetscore, workflowid, wholeid, partd_key
)
SELECT
    _id, calldate, strparams, productname, entryid, finaldealtype, entrystatus, invokemode, workflowtype, policyset, statdate, cost, test, productid, customparams, appname, workflowname, entryparams, entryoccurtime, token, workflowcode, addtype, productcode, recordtime, finaldealtypename, longparams, entryparamsser, workflowversion, thirdservice, policysetscore, workflowid, wholeid,
    TO_CHAR(TO_TIMESTAMP(recordtime / 1000), 'YYYYMM') AS partd_key
FROM atreus_es.atreus_entrysearch
WHERE recordtime >= 1744041600000 AND recordtime < 1744074000000; -- 限制时间范围

insert into atreus_es.atreus_entrysearch_202504(
    _id, calldate, strparams, productname, entryid, finaldealtype, entrystatus, invokemode, workflowtype, policyset, statdate, cost, test, productid, customparams, appname, workflowname, entryparams, entryoccurtime, token, workflowcode, addtype, productcode, recordtime, finaldealtypename, longparams, entryparamsser, workflowversion, thirdservice, policysetscore, workflowid, wholeid, partd_key
)
SELECT
    _id, calldate, strparams, productname, entryid, finaldealtype, entrystatus, invokemode, workflowtype, policyset, statdate, cost, test, productid, customparams, appname, workflowname, entryparams, entryoccurtime, token, workflowcode, addtype, productcode, recordtime, finaldealtypename, longparams, entryparamsser, workflowversion, thirdservice, policysetscore, workflowid, wholeid,
    TO_CHAR(TO_TIMESTAMP(recordtime / 1000), 'YYYYMM') AS partd_key
FROM atreus_es.atreus_entrysearch
WHERE recordtime >= 1744041600000 AND recordtime < 1744074000000; -- 限制时间范围


select * from atreus_es.atreus_entrysearch_partd;


SELECT * FROM atreus_es.freyr_partd;




BEGIN;

/*
DROP TABLE atreus_es.freyr_partd;
*/

-- Type: TABLE ; Name: freyr_partd; Owner: p4_201351809199932519

CREATE TABLE atreus_es.freyr_partd (
    _id text NOT NULL,
    msg text,
    invoketype integer,
    cost integer,
    tokenid text,
    docid text,
    errorcode text,
    responsedata json,
    servicedisplayname text,
    requestparam json,
    servicename text,
    sequenceid text,
    businessservicename text,
    requesttime bigint,
    businesscode text,
    processdata json,
    recordtime bigint,
    organizationcode text,
    success boolean,
    createdtime text,
    status integer,
    partd_key text NOT NULL
    ,PRIMARY KEY (_id, partd_key)
)
PARTITION BY LIST (partd_key) -- 使用 LIST 分区
with (
orientation = 'column',
storage_format = 'orc',
bitmap_columns = '_id,msg,tokenid,docid,errorcode,servicedisplayname,servicename,sequenceid,businessservicename,businesscode,organizationcode,createdtime',
dictionary_encoding_columns = '_id:auto,msg:auto,tokenid:auto,docid:auto,errorcode:auto,servicedisplayname:auto,servicename:auto,sequenceid:auto,businessservicename:auto,businesscode:auto,organizationcode:auto,createdtime:auto,partd_key:auto',
distribution_key = '_id',
table_group = 'data_dws_tg_default',
table_storage_mode = 'hot',
time_to_live_in_seconds = '3153600000'
);



COMMENT ON TABLE atreus_es.freyr_partd IS NULL;
ALTER TABLE atreus_es.freyr_partd OWNER TO p4_201351809199932519;


END;

CREATE TABLE atreus_es.freyr_partd_202503 PARTITION OF atreus_es.freyr_partd
FOR VALUES IN ('202503');
CREATE TABLE atreus_es.freyr_partd_202504 PARTITION OF atreus_es.freyr_partd
FOR VALUES IN ('202504');

CREATE TABLE atreus_es.freyr_partd_202505 PARTITION OF atreus_es.freyr_partd
FOR VALUES IN ('202505');
CREATE TABLE atreus_es.freyr_partd_202506 PARTITION OF atreus_es.freyr_partd
FOR VALUES IN ('202506');
CREATE TABLE atreus_es.freyr_partd_202507 PARTITION OF atreus_es.freyr_partd
FOR VALUES IN ('202507');
CREATE TABLE atreus_es.freyr_partd_202508 PARTITION OF atreus_es.freyr_partd
FOR VALUES IN ('202508');



select * from atreus_es.freyr_partd_202503;

select * from atreus_es.freyr_partd;

select count(*) from atreus_es.freyr_partd;

select distinct risk_channel_code, case when risk_channel_code='TYH_APPZY_TYH_APPZY' then 'TYH_APPZY'  when risk_channel_code='TYH_RP' then 'TYH_RP' else channel_code end as channel_code from zl_data_clean.dim_zl_channel_info ;where risk_channel_code = 'RS_RL';



BEGIN;

/*
DROP TABLE atreus_es.atreus_componentlog_partd;
*/

-- Type: TABLE ; Name: atreus_componentlog_partd; Owner: p4_201351809199932519

CREATE TABLE atreus_es.atreus_componentlog_partd (
    _id text NOT NULL,
    nodename text,
    tasksuccess boolean,
    reqindexs text,
    statdate bigint,
    reqparams text,
    nodecost integer,
    resparams text,
    nodetype text,
    entryid text,
    token text,
    workflowcode text,
    incomingflow text,
    workflowversion integer,
    nodeid text,
    calltime text,
    partd_key text NOT NULL
    ,PRIMARY KEY (_id, partd_key)
) PARTITION BY LIST (partd_key)with (
orientation = 'column',
storage_format = 'orc',
bitmap_columns = '_id,nodename,reqindexs,reqparams,resparams,nodetype,entryid,token,workflowcode,incomingflow,nodeid,calltime',
dictionary_encoding_columns = '_id:auto,nodename:auto,reqindexs:auto,reqparams:auto,resparams:auto,nodetype:auto,entryid:auto,token:auto,workflowcode:auto,incomingflow:auto,nodeid:auto,calltime:auto,partd_key:auto',
distribution_key = '_id',
table_group = 'data_dws_tg_default',
table_storage_mode = 'hot',
time_to_live_in_seconds = '3153600000'
);



COMMENT ON TABLE atreus_es.atreus_componentlog_partd IS NULL;
ALTER TABLE atreus_es.atreus_componentlog_partd OWNER TO p4_201351809199932519;


END;

CREATE TABLE atreus_es.atreus_componentlog_202505 PARTITION OF atreus_es.atreus_componentlog_partd
FOR VALUES IN ('202505');
CREATE TABLE atreus_es.atreus_componentlog_202506 PARTITION OF atreus_es.atreus_componentlog_partd
FOR VALUES IN ('202506');
CREATE TABLE atreus_es.atreus_componentlog_202507 PARTITION OF atreus_es.atreus_componentlog_partd
FOR VALUES IN ('202507');
CREATE TABLE atreus_es.atreus_componentlog_202508 PARTITION OF atreus_es.atreus_componentlog_partd
FOR VALUES IN ('202508');

BEGIN;

/*
DROP TABLE atreus_es.atreus_entryinvokeresult_partd;
*/

-- Type: TABLE ; Name: atreus_entryinvokeresult_partd; Owner: p4_201351809199932519

CREATE TABLE atreus_es.atreus_entryinvokeresult_partd (
    _id text NOT NULL,
    cost integer,
    test integer,
    servicename text,
    version integer,
    entryid text,
    token text,
    workflowcode text,
    productcode text,
    finaldealtypecode text,
    envoutputs text,
    invokemode integer,
    guardid text,
    finaldealtypename text,
    reasoncode integer,
    calltime bigint,
    reasondesc text,
    partd_key text NOT NULL
    ,PRIMARY KEY (_id, partd_key)
)  PARTITION BY LIST (partd_key)with (
orientation = 'column',
storage_format = 'orc',
bitmap_columns = '_id,servicename,entryid,token,workflowcode,productcode,finaldealtypecode,envoutputs,guardid,finaldealtypename,reasondesc',
dictionary_encoding_columns = '_id:auto,servicename:auto,entryid:auto,token:auto,workflowcode:auto,productcode:auto,finaldealtypecode:auto,envoutputs:auto,guardid:auto,finaldealtypename:auto,reasondesc:auto,partd_key:auto',
distribution_key = '_id',
table_group = 'data_dws_tg_default',
table_storage_mode = 'hot',
time_to_live_in_seconds = '3153600000'
);



COMMENT ON TABLE atreus_es.atreus_entryinvokeresult_partd IS NULL;
ALTER TABLE atreus_es.atreus_entryinvokeresult_partd OWNER TO p4_201351809199932519;

CREATE TABLE atreus_es.atreus_entryinvokeresult_202504 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202504');
CREATE TABLE atreus_es.atreus_entryinvokeresult_202503 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202503');
CREATE TABLE atreus_es.atreus_entryinvokeresult_202502 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202502');
END;
CREATE TABLE atreus_es.atreus_entryinvokeresult_202504 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202504');
CREATE TABLE atreus_es.atreus_entryinvokeresult_202503 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202503');
CREATE TABLE atreus_es.atreus_entryinvokeresult_202502 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202502');


CREATE TABLE atreus_es.atreus_entryinvokeresult_202505 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202505');
CREATE TABLE atreus_es.atreus_entryinvokeresult_202506 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202506');
CREATE TABLE atreus_es.atreus_entryinvokeresult_202507 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202507');
CREATE TABLE atreus_es.atreus_entryinvokeresult_202508 PARTITION OF atreus_es.atreus_entryinvokeresult_partd
FOR VALUES IN ('202508');



select * from realtime_dws.dws_risk_idno_overdue_d_v;





SELECT * from zl_data_clean.atreus_entrysearch_longparams;
SELECT * FROM zl_data_clean.derivatives_v2_ex;

select * from atreus_es.atreus_componentlog_partd;
select count(*) from atreus_es.atreus_componentlog_partd;
select * from atreus_es.atreus_componentlog_202502;
-- DELETE FROM atreus_es.atreus_componentlog_202502;
select * from atreus_es.atreus_componentlog_202503;



select * from atreus_es.atreus_entryinvokeresult_partd where token='tokene81e3f8caa4a4324a3461ed427ce1f4a-1740672000000';
-- DELETE FROM atreus_es.atreus_entryinvokeresult_202502;
select * from atreus_es.atreus_entryinvokeresult_partd where calltime>1740712448000;
select count(*) from atreus_es.atreus_entryinvokeresult_partd;
select * from atreus_es.atreus_entryinvokeresult_partd;

select * from atreus_es.atreus_entryinvokeresult_100;

select * from atreus_es.atreus_entrysearch;
select * from atreus_es.atreus_componentlog_202502;


select * from mongo_ods.tanzhi_qyfxzs ORDER BY create_time DESC ;

select count(*) from zl_fk_data.d0_billing_date_case_remind_status where d0_status_date='20250414' and status=1;
select count(*) from zl_fk_data.d0_billing_date_case_remind_status where d0_status_date='20250413' and status=1;


select   t3.risk_business_id
FROM    mongo_ods.fulin_online_status_coll t1
LEFT JOIN mongo_ods.fulin_online_status_map t2
ON      t1.req_id = t2.fulin_online_status_req_id
LEFT JOIN mongo_ods.fulin_online_status_req t3
ON      t2.customer_request_id = t3.customer_request_id
where t3.risk_business_id ='1912056811310321664';

select * from mongo_ods.fulin_online_status_coll where req_id='4992e577-7597-45d1-bac5-d1d75aea3ecf';
select * from mongo_ods.fulin_online_status_req where customer_request_id='1980565f-5e1d-44ab-8009-f8f2a5c1a823';
select * from mongo_ods.fulin_online_status_map where customer_request_id='1980565f-5e1d-44ab-8009-f8f2a5c1a823';
select * from mongo_ods.fulin_online_status_req where risk_business_id='1912056811310321664';

select * from mongo_ods.fulin_online_status_map_202504171448 where create_time>'2025-04-15';

select * from mongo_ods.fulin_online_status_map where create_time>'2025-04-10';
select * from mongo_ods.fulin_online_status_map where create_time>'2023-04-11' and is_cache=True;
select * from mongo_ods.fulin_online_status_map where create_time>'2025-04-17 14:1' and is_cache=True;

select * from mongo_ods.fl_3factor_upg_map where create_time>'2025-04-17 09:15:45.498166';
select * from mongo_ods.fl_3factor_upg_map where create_time>'2025-04-11' and is_cache=True;
select * from mongo_ods.fl_3factor_upg_map where req_id is NULL ;


select * from mongo_ods.mayi_qyfxzs_v2;


SELECT * FROM mongo_ods.ml_models_212
where model_name='score_ccard_V2_20250415'

SELECT * FROM mongo_ods.ml_models_212;
SELECT * FROM mongo_ods.ml_models;

select * from mongo_ods.check_user_query_records ORDER BY create_time DESC ;

select * from mongo_ods.model_scoring_records order by update_time desc;

delete from atreus_es.atreus_entrysearch_partd;
delete from atreus_es.atreus_entrysearch_202504;
delete from atreus_es.atreus_entrysearch_202501;

select * from atreus_es.atreus_entrysearch_partd;
-- record_time_1 = '1744214400000'  # 2025-04-10 00:00:00
-- record_time_2= '1744992000000'  # 2025-04-19 00:00:00
select count(*) from atreus_es.atreus_componentlog_202504;

select count(*) from atreus_es.atreus_componentlog_202504;


delete from atreus_es.atreus_entrysearch_partd;

select TO_TIMESTAMP(recordtime/1000), workflowcode,* from  atreus_es.atreus_entrysearch_partd where entryid='1914154553255956480';

select TO_TIMESTAMP(recordtime/1000), workflowcode,* from  atreus_es.atreus_entrysearch where entryid='1914154553255956480';



select TO_TIMESTAMP(recordtime/1000), workflowcode,* from  atreus_es.atreus_entrysearch_partd where recordtime<=1745204983435 order by recordtime DESC;
select TO_TIMESTAMP(recordtime/1000), workflowcode,* from  atreus_es.atreus_entrysearch_partd where recordtime>=1745204983435 order by recordtime asc;


select TO_TIMESTAMP(recordtime/1000), workflowcode,* from  atreus_es.atreus_entrysearch_partd where entryid in ('1914154553255956480','da83b9c4-f8ef-4c81-86e6-40396bb85b98');
select TO_TIMESTAMP(recordtime/1000), workflowcode,* from  atreus_es.atreus_entrysearch where entryid in('1914154553255956480','da83b9c4-f8ef-4c81-86e6-40396bb85b98');

select TO_TIMESTAMP(recordtime/1000), workflowcode,* from  atreus_es.atreus_entrysearch_partd order by recordtime asc;


select  t1.token,TO_TIMESTAMP(t1.recordtime/1000),t2.token from (
select token,recordtime from atreus_es.atreus_entrysearch_partd
where substr(TO_TIMESTAMP(recordtime/1000)::text,1,13)='2025-04-21 14' ) t1
left join (select token from atreus_es.atreus_entrysearch
where substr(TO_TIMESTAMP(recordtime/1000)::text,1,13)='2025-04-21 14') t2 on t1.token=t2.token
where t2.token IS  null;

select  t1.token,TO_TIMESTAMP(t1.recordtime/1000),t2.token from (
select token,recordtime from atreus_es.atreus_entrysearch
where substr(TO_TIMESTAMP(recordtime/1000)::text,1,13)='2025-04-21 14' ) t1
left join (select token from atreus_es.atreus_entrysearch_partd
where substr(TO_TIMESTAMP(recordtime/1000)::text,1,13)='2025-04-21 14') t2 on t1.token=t2.token
where t2.token IS  null;


select * from realtime_dws.dws_risk_idno_overdue_d_v;

