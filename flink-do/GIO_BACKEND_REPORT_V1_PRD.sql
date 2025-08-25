--********************************************************************--
-- Author:         shidexiao
-- Created Time:   2025-03-25 15:01:09
-- Description:    Write your description here
-- Hints:          You can use SET statements to modify the configuration
--********************************************************************--


-- 撞库
-- userCollisionRequestResult 用户撞库结果
-- zws_middleware.zl_decision_info_log 核心决策结果信息表

CREATE TEMPORARY TABLE if not EXISTS zl_decision_info_log_source
(
    id                    bigint NOT NULL
    ,success              VARCHAR
    ,uuid                 VARCHAR
    ,channel_code         VARCHAR
    ,md5_value            VARCHAR
    ,final_deal_type_code VARCHAR
    ,final_deal_type_name VARCHAR
    ,refuse_code          VARCHAR
    ,reason_code          VARCHAR
    ,reason_desc          VARCHAR
    ,del_flag             VARCHAR
    ,create_by            VARCHAR
    ,create_time          TIMESTAMP
    ,update_by            VARCHAR
    ,update_time          TIMESTAMP
    ,mode                 VARCHAR
    ,phone_md5            VARCHAR
    ,id_no_md5            VARCHAR
    ,refuse_msg           VARCHAR
    ,-- ,hg_binlog_lsn          BIGINT
    hg_binlog_event_type BIGINT
    -- ,hg_binlog_timestamp_us BIGINT
    -- ,event_time timestamp(3)
    -- ,WATERMARK FOR create_time AS create_time - INTERVAL '10' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zl_decision_info_log'
    ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

CREATE TEMPORARY TABLE zx_credit_user_info_source
(
    id_card_no    VARCHAR
    ,customer_no  VARCHAR
    ,user_id      VARCHAR
    ,name         VARCHAR
    ,applicant_id BIGINT
    ,mobile       VARCHAR
    ,create_time  TIMESTAMP
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zx_credit_user_info'
    ,'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
    ,'cache' = 'LRU'
    ,'cacheTTLMs' = '300000'
)
;

CREATE TEMPORARY TABLE dim_channel_product_mapping
(
    channel_code  VARCHAR
    ,product_code VARCHAR
    ,source_code  VARCHAR
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zl_fk_data.temp_channel_product_mapping2'
    ,'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
    ,'cache' = 'LRU'
    ,'cacheTTLMs' = '300000'
)
;

CREATE TEMPORARY TABLE product_credit_amt_detail_source
(
    customer_no    VARCHAR
    ,product_code  VARCHAR
    ,credit_amount DECIMAL(12, 2)
    ,used_amount   DECIMAL(12, 2)
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'amt_center.product_credit_amt_detail'
    ,'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
    -- ,'cache' = 'LRU'
    -- ,'cacheTTLMs' = '300000'
)
;

CREATE TEMPORARY TABLE gio_event_message_sink
(
    event_time                 BIGINT NOT NULL
    ,-- 事件时间
    event_key                 VARCHAR NOT NULL
    ,-- 埋点事件标识
    anonymous_id              VARCHAR
    ,-- 访问用户ID
    login_user_key            VARCHAR
    ,-- 登录用户KEY
    login_user_id             VARCHAR
    ,-- 登录用户ID
    result_var                VARCHAR
    ,-- 结果
    reason_var                VARCHAR
    ,-- 原因
    login_method_var          VARCHAR
    ,-- 登录方式
    supplier_var              VARCHAR
    ,-- 认证供应商
    time_duration_var         VARCHAR
    ,-- 处理时长
    credit_limit              DOUBLE
    ,-- 授信额度
    available_limit           DOUBLE
    ,-- 可用额度
    credit_type_var           VARCHAR
    ,-- 授信类型
    bank_name_var             VARCHAR
    ,-- 银行卡归属行
    service_provider_name_var VARCHAR
    ,-- 绑卡服务商
    approval_duration_var     VARCHAR
    ,-- 审批时长
    capital_limit             DOUBLE
    ,-- 资方额度
    capital_credit_var        VARCHAR
    ,-- 审核资方
    withdraw_limit            DOUBLE
    ,-- 放款额度
    capital_name_var          VARCHAR
    ,-- 放款资方
    stage                     INT
    ,-- 期数
    price                     DOUBLE
    ,-- 金额
    repay_statement_var       VARCHAR
    ,-- 还款状态
    repay_method_var          VARCHAR
    ,-- 还款方式
    traffic_channel_var       VARCHAR
    ,user_source_var           VARCHAR
    ,report_status             VARCHAR
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'realtime.gio_event_message_prd'
    -- ,'mutatetype' = 'insertorreplace'
    -- ,'ignoredelete' = 'true'
    -- ,'sink.delete-strategy' = 'DELETE_ROW_ON_PK'
    -- ,'partial-insert.enabled' = 'true'
    -- ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

CREATE TEMPORARY VIEW decision_user_view
AS SELECT
    t1.*
    ,t3.source_code AS user_source_var
    ,t3.product_code
    ,s.customer_no
    -- ,t2.user_id
    ,s.mobile
    ,s.name
    ,s.id_card_no
    --, t2.applicant_id
FROM zl_decision_info_log_source t1

-- LEFT JOIN zx_credit_user_info_source FOR SYSTEM_TIME AS OF PROCTIME() as t2
--     ON t1.id_no_md5 =  MD5(t2.id_card_no)

-- LEFT JOIN LATERAL
--     (SELECT customer_no,mobile, name, id_card_no FROM zx_credit_user_info_source as t2 WHERE t1.id_no_md5 =  MD5(t2.id_card_no)) s ON TRUE
    LEFT JOIN LATERAL
    (
    SELECT
        t.customer_no
        ,t.mobile
        ,t.name
        ,t.id_card_no
    from (
        SELECT
            t2.customer_no
            ,t2.mobile
            ,t2.name
            ,t2.id_card_no
            ,t2.create_time
            ,row_number()
            OVER (
                partition by t2.id_card_no
                ORDER BY  t2.create_time DESC
            ) rk
        FROM zx_credit_user_info_source as t2
        WHERE MD5(t2.id_card_no) = t1.id_no_md5
            and customer_no is not NULL
            and mobile is not NULL
    ) t
    WHERE rk = 1
) s
        ON TRUE
    LEFT JOIN dim_channel_product_mapping FOR SYSTEM_TIME AS OF PROCTIME() as t3
        ON t1.channel_code = t3.channel_code
-- where t2.customer_no is not NULL
;

CREATE TEMPORARY VIEW __viewName_1
AS SELECT
    t1.*
    ,case
        when t1.final_deal_type_code = 'Accept' THEN '成功'
        when t1.final_deal_type_code = 'Reject' THEN '失败'
    END as result_var
    ,t2.credit_amount   -- APP产品授信额度
    ,t2.used_amount   -- App产品已用额度
    ,(
        t2.credit_amount - t2.used_amount) as available_limit -- 产品域可用额度
FROM decision_user_view as t1
    LEFT JOIN product_credit_amt_detail_source FOR SYSTEM_TIME AS OF PROCTIME() as t2
        ON t1.customer_no = t2.customer_no
        and t2.product_code = t1.product_code
WHERE t1.mobile is not NULL
;


-- withdrawalRiskControlResult	风控借款审核结果返回

--  zws_middleware.zl_drms_judgements 众利-风控决策-提交申请

CREATE TEMPORARY TABLE if not EXISTS zy_zl_drms_judgements_source
(
    id                            bigint NOT NULL
    ,apply_id                     VARCHAR
    ,apply_type                   VARCHAR
    ,apply_principal              numeric(38, 18)
    ,apply_term                   bigint
    ,apply_loan_count             bigint
    ,user_id                      VARCHAR
    ,user_idcard_number           VARCHAR
    ,user_mobile_number           VARCHAR
    ,user_emergency_contact_list  VARCHAR
    ,user_name                    VARCHAR
    ,user_source                  VARCHAR
    ,partner_code                 VARCHAR
    ,notify_url                   VARCHAR
    ,business_number              VARCHAR
    ,user_success_credit_time     VARCHAR
    ,user_credit_amount_total     VARCHAR
    ,user_credit_amount_available VARCHAR
    ,check_result                 VARCHAR
    ,credit_amount                VARCHAR
    ,channel_code                 VARCHAR
    ,del_flag                     VARCHAR
    ,create_time                  timestamp
    ,update_time                  timestamp
    ,create_by_id                 bigint
    ,update_by_id                 bigint
    ,apply_error_msg              VARCHAR
    ,funds_code                   VARCHAR
    ,product_code                 VARCHAR
    ,funds_product_code           VARCHAR
    -- ,hg_binlog_lsn          BIGINT
    ,hg_binlog_event_type         BIGINT
    -- ,hg_binlog_timestamp_us BIGINT
    -- ,event_time timestamp(3)
    -- ,WATERMARK FOR create_time AS create_time - INTERVAL '10' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zl_drms_judgements'
    ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

-- zws_middleware.zx_credit_info 360智信授信申请_返回结果的授信信息

CREATE TEMPORARY TABLE zx_credit_info_source
(
    id                   BIGINT
    ,credit_time         VARCHAR
    ,credit_amt          VARCHAR
    ,used_amt            VARCHAR
    ,credit_type         VARCHAR
    ,status              VARCHAR
    -- ,user_group           VARCHAR
    -- ,begin_date           VARCHAR
    -- ,end_date             VARCHAR
    ,user_id             VARCHAR
    ,applicant_result_id BIGINT
    ,create_by           VARCHAR
    -- ,update_by            VARCHAR
    -- ,create_time          timestamp(3)
    -- ,update_time          timestamp
    ,del_flag            VARCHAR
    -- ,channel_code         VARCHAR
    -- ,funds_code           VARCHAR
    -- ,freeze_recovery_time VARCHAR
    -- ,customer_no          VARCHAR
    -- ,event_time timestamp(3)
        -- ,hg_binlog_lsn          BIGINT
    -- ,hg_binlog_event_type BIGINT
    -- ,hg_binlog_timestamp_us BIGINT
    -- ,WATERMARK FOR create_time AS create_time - INTERVAL '20' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zx_credit_info'
    ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;


--  zws_middleware.zl_drms_judgements 众利-风控决策-提交申请

CREATE TEMPORARY TABLE if not EXISTS zx_credit_applicant_result_source
(
    id        bigint NOT NULL
    ,apply_id VARCHAR
    -- ,hg_binlog_lsn          BIGINT
    -- ,hg_binlog_event_type   BIGINT
    -- ,hg_binlog_timestamp_us BIGINT
    -- ,event_time timestamp(3)
    -- ,WATERMARK FOR create_time AS create_time - INTERVAL '10' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zx_credit_applicant_result'
    ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

CREATE TEMPORARY VIEW withdrawal_risk_control_result
AS SELECT
    user_mobile_number AS login_user_id
    ,CASE
        WHEN check_result = 'pass' THEN '成功'
        WHEN check_result = 'reject' THEN '失败'
    END AS result_var
    ,'' AS reason_var
    ,CAST(user_credit_amount_total AS DOUBLE) AS credit_limit
    ,CAST(user_credit_amount_total AS VARCHAR) AS credit_limit_str
    ,CAST(user_credit_amount_available AS DOUBLE) AS available_limit
    ,CAST(user_credit_amount_available AS VARCHAR) AS available_limit_str
    ,channel_code AS traffic_channel_var
    ,CASE
        WHEN channel_code IN (
            'APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
            'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC'
        ) THEN '众利'
        WHEN channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY') THEN '龙力花'
        WHEN channel_code IN (
            'LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
            'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
            'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT'
        ) THEN '天源花'
    END AS user_source_var
FROM zy_zl_drms_judgements_source
WHERE apply_type <> 'open_card'
    and check_result IS NOT NULL
;

CREATE TEMPORARY VIEW credit_risk_control_result
AS SELECT
    -- 第一条 SQL 中的字段选取和计算逻辑
    -- j.*,
    d.mobile as login_user_id
    ,CASE
        WHEN j.check_result = 'pass' THEN '成功'
        WHEN j.check_result = 'reject' THEN '失败'
    END AS result_var
    ,'' AS reason_var
    ,j.channel_code AS traffic_channel_var
    ,CASE
        WHEN j.channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC') THEN '众利'
        WHEN j.channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY') THEN '龙力花'
        WHEN j.channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT') THEN '天源花'
    END AS user_source_var
    ,'风控授信' AS credit_type_var
    ,EXTRACT(EPOCH FROM j.update_time) - EXTRACT(EPOCH FROM j.create_time) AS time_duration_var
    ,-- 第二条 SQL 中的字段选取和计算逻辑
    CAST(c.credit_amt AS DOUBLE) as credit_limit
    ,c.credit_amt AS credit_limit_str
    ,CAST(ROUND(
        CAST(c.credit_amt AS NUMERIC) - CAST(c.used_amt AS NUMERIC)
        ,2
    ) AS DECIMAL(10, 2)) AS available_limit
    ,CAST(ROUND(
        CAST(c.credit_amt AS NUMERIC) - CAST(c.used_amt AS NUMERIC)
        ,2
    ) AS VARCHAR) AS available_limit_str
FROM zy_zl_drms_judgements_source j
    LEFT JOIN zx_credit_applicant_result_source FOR SYSTEM_TIME AS OF PROCTIME() r
        ON
    j.apply_id = r.apply_id
    LEFT JOIN zx_credit_info_source FOR SYSTEM_TIME AS OF PROCTIME() c
        ON
    c.applicant_result_id = r.id
    LEFT JOIN zx_credit_user_info_source FOR SYSTEM_TIME AS OF PROCTIME() d
        ON
    j.user_id = d.user_id
WHERE j.apply_type = 'open_card'
    AND j.check_result IS NOT NULL
;

BEGIN STATEMENT SET
;

-- withdrawalRiskControlResult

INSERT INTO gio_event_message_sink (
    event_time
    ,event_key
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'withdrawalRiskControlResult' as event_key
    ,'' as anonymous_id
    ,'' as login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,GrowingEventZ_PRD(
        login_user_id
        ,CAST('withdrawalRiskControlResult' AS VARCHAR)
        ,MAP['result_var', result_var,
            'reason_var', reason_var,
            'creditLimit', credit_limit_str,
            'availableLimit', available_limit_str,
            'trafficChannel_var',traffic_channel_var,
            'userSource_var',user_source_var]
    ) AS report_status
from withdrawal_risk_control_result
where user_source_var = '众利'
;

INSERT INTO gio_event_message_sink (
    event_time
    ,event_key
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'withdrawalRiskControlResult' as event_key
    ,'' as anonymous_id
    ,'' as login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,GrowingEventH_PRD(
        login_user_id
        ,CAST('withdrawalRiskControlResult' AS VARCHAR)
        ,MAP['result_var', result_var,
            'reason_var', reason_var,
            'creditLimit', credit_limit_str,
            'availableLimit', available_limit_str,
            'trafficChannel_var',traffic_channel_var,
            'userSource_var',user_source_var]
    ) AS report_status
from withdrawal_risk_control_result
where user_source_var IN ('龙力花','天源花')
;


-- creditRiskControlResult

INSERT INTO gio_event_message_sink (
    event_time
    ,event_key
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_type_var
    ,time_duration_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'creditRiskControlResult' as event_key
    ,'' as anonymous_id
    ,'$basic_userId' as login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_type_var
    ,CAST(time_duration_var AS VARCHAR) as time_duration_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,GrowingEventZ_PRD(
        login_user_id
        ,CAST('creditRiskControlResult' AS VARCHAR)
        ,MAP['result_var', result_var,
            'reason_var', reason_var,
            'creditLimit', credit_limit_str,
            'availableLimit', available_limit_str,
            'trafficChannel_var',traffic_channel_var,
            'timeDuration_var',CAST(time_duration_var AS VARCHAR),
            'creditType_var',credit_type_var,
            'userSource_var',user_source_var]
    ) AS report_status
FROM credit_risk_control_result
where user_source_var = '众利'
;

INSERT INTO gio_event_message_sink (
    event_time
    ,event_key
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_type_var
    ,time_duration_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'creditRiskControlResult' as event_key
    ,'' as anonymous_id
    ,'$basic_userId' as login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_type_var
    ,CAST(time_duration_var AS VARCHAR) as time_duration_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,GrowingEventH_PRD(
        login_user_id
        ,CAST('creditRiskControlResult' AS VARCHAR)
        ,MAP['result_var', result_var,
            'reason_var', reason_var,
            'creditLimit', credit_limit_str,
            'availableLimit', available_limit_str,
            'trafficChannel_var',traffic_channel_var,
            'timeDuration_var',CAST(time_duration_var AS VARCHAR),
            'creditType_var',credit_type_var,
            'userSource_var',user_source_var]
    ) AS report_status
FROM credit_risk_control_result
where user_source_var IN ('龙力花','天源花')
;

INSERT INTO gio_event_message_sink (
    event_time
    ,event_key
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,result_var
    ,reason_var
    ,credit_limit
    ,available_limit
    ,traffic_channel_var
    ,user_source_var
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'userCollisionRequestResult' as event_key
    ,'' as anonymous_id
    ,'$basic_userId' as login_user_key
    ,mobile as login_user_id
    ,result_var
    ,refuse_msg AS reason_var
    ,credit_amount AS credit_limit
    ,available_limit
    ,channel_code AS traffic_channel_var
    ,user_source_var
    ,GrowingEventF_PRD(
        user_source_var
        ,mobile
        ,CAST('userCollisionRequestResult' AS VARCHAR)
        ,MAP['result_var', result_var,
            'reason_var', refuse_msg,
            'creditLimit', CAST(credit_amount AS VARCHAR),
            'availableLimit', CAST(available_limit AS VARCHAR),
            'trafficChannel_var',channel_code,
            'userSource_var',user_source_var]
    ) AS report_status
FROM __viewName_1
;


END
;


-- select * from __viewName_1;



