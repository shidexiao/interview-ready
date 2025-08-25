--********************************************************************--
-- Author:         shidexiao
-- Created Time:   2025-03-25 15:01:32
-- Description:    Write your description here
-- Hints:          You can use SET statements to modify the configuration
--********************************************************************--

CREATE TEMPORARY TABLE if not EXISTS dwd_user_info_real_source
(
    user_id               VARCHAR NOT NULL
    ,channel_key          VARCHAR
    ,cust_no              VARCHAR
    ,user_name            VARCHAR
    ,id_no                VARCHAR
    ,mobile_no            VARCHAR
    ,marital_status       VARCHAR
    ,education            VARCHAR
    ,school               VARCHAR
    ,profession           VARCHAR
    ,job_type             VARCHAR
    ,job_position         VARCHAR
    ,income               VARCHAR
    ,industry             VARCHAR
    ,province_code        VARCHAR
    ,city_code            VARCHAR
    ,district_code        VARCHAR
    ,address              VARCHAR
    ,positive             VARCHAR
    ,negative             VARCHAR
    ,begin_time_ocr       VARCHAR
    ,duetime_ocr          VARCHAR
    ,address_ocr          VARCHAR
    ,sex_ocr              VARCHAR
    ,ethnic_ocr           VARCHAR
    ,issue_ocr            VARCHAR
    ,event_time           timestamp(3)
    ,create_time          timestamp(3)
    ,update_time          timestamp(3)
    -- ,PRIMARY KEY (user_id)  not enforced
            -- ,hg_binlog_lsn          BIGINT
    ,hg_binlog_event_type BIGINT
    -- ,hg_binlog_timestamp_us BIGINT
    -- ,WATERMARK FOR hg_binlog_timestamp_us AS hg_binlog_timestamp_us - INTERVAL '20' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    -- ,'dbname' = 'data_realtime'
    ,'dbname' = '${secret_values.holo_dbname_data_realtime}'
    ,'tablename' = 'realtime_dwd.dwd_user_info_real'
    ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

CREATE TEMPORARY TABLE if not EXISTS zx_credit_user_info_source
(
    user_id       VARCHAR NOT NULL
    ,channel_code VARCHAR
    ,mobile       VARCHAR
    ,customer_no  VARCHAR
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zx_credit_user_info'
    ,'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

CREATE TEMPORARY TABLE if not EXISTS zx_credit_applicant_result_source
(
    id           bigint NOT NULL
    ,apply_id    VARCHAR
    ,status      VARCHAR
    ,customer_no VARCHAR
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

CREATE TEMPORARY TABLE if not EXISTS zl_area_code_source
(
    region_code        VARCHAR
    ,region_short_name VARCHAR
    ,region_name       VARCHAR

    -- ,WATERMARK FOR hg_binlog_timestamp_us AS hg_binlog_timestamp_us - INTERVAL '20' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zl_area_code'
    ,'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

-- 额度中心 binlog

CREATE TEMPORARY TABLE product_credit_amt_detail_source2
(
    customer_no           VARCHAR
    ,product_code         VARCHAR
    ,credit_amount        DECIMAL(16, 4)
    ,used_amount          DECIMAL(16, 4)
    ,status               VARCHAR -- 授信状态NR-正常，TF-临时冻结,只有这两个值
    -- ,hg_binlog_lsn          BIGINT
    ,hg_binlog_event_type BIGINT
    -- ,hg_binlog_timestamp_us BIGINT
    -- ,WATERMARK FOR create_time AS create_time - INTERVAL '10' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'amt_center.product_credit_amt_detail'
    ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;



-- 额度中心

CREATE TEMPORARY TABLE product_credit_amt_detail_source
(
    customer_no    VARCHAR
    ,product_code  VARCHAR
    ,credit_amount DECIMAL(16, 4)
    ,used_amount   DECIMAL(16, 4)
    ,status        VARCHAR -- 授信状态NR-正常，TF-临时冻结,只有这两个值
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

CREATE TEMPORARY VIEW user_info
AS SELECT
    mobile_no as basic_mobile  --用户手机号
    ,SUBSTRING(user_name, 1, 1) as lastName_ppl      --姓
    ,id_no    --身份证号码
    ,TO_DATE(SUBSTRING(id_no, 7, 8), 'yyyyMMdd') AS birth_date --年龄
    ,channel_key
    ,cust_no as customer_no
    ,CASE
        WHEN channel_key IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC') THEN '众利'
        WHEN channel_key IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY') THEN '龙力花'
        WHEN channel_key IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT') THEN '天源花'
    END AS user_source_var
    ,CASE
        WHEN channel_key IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC') THEN 'ZL_APP'
        WHEN channel_key IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY') THEN 'LLH_APP'
        WHEN channel_key IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT') THEN 'TYH_APP'
    END AS product_code
    ,case sex_ocr
        WHEN 'M' THEN '男'
        WHEN 'F' THEN '女'
        ELSE '未知'
    END as gender_ppl   --性别
    ,case marital_status
        when '10' then '未婚'
        when '20' then '已婚'
        WHEN '30' THEN '丧偶'
        WHEN '40' THEN '离婚'
        ELSE '未知'
    end as marriage_ppl --婚姻
    ,case education
        when '10' then '研究生'
        when '20' then '大学本科'
        when '30' then '大专'
        when '60' then '高中'
        when '70' then '初中'
        when '80' then '小学'
        when '99' then '未知'
        ELSE '未知'
    end as education_ppl  --学历
    ,city_code--居住地_市
    ,case income
        when '1' then '5000以下'
        when '2' then '5001-10000'
        when '3' then '10001-20000'
        when '4' then '20001以上'
        ELSE '未知'
    end as income_ppl --月收入
    ,case industry
        when 'G' then '信息传输、计算机服务和软件业'
        when 'J' then '金融业'
        when 'E' then '建筑业'
        when 'K' then '房地产业'
        when 'H' then '批发和零售业'
        when 'P' then '教育'
        when 'R' then '文化、体育和娱乐业'
        when 'Q' then '卫生、社会保障和社会福利业'
        when 'D' then '电力、燃气及水的生产和供应业'
        ELSE '未知'
    end as industry_ppl--行业
    ,CASE profession
        WHEN '0' THEN '国家机关、党群组织、企业、事业单位负责人'
        WHEN '1' THEN '专业技术人员'
        WHEN '3' THEN '办事人员和有关人员'
        WHEN '4' THEN '商业、服务业人员'
        WHEN '5' THEN '农、林、牧、渔、水利业生产人员'
        WHEN '6' THEN '生产、运输设备操作人员及有关人员'
        WHEN 'X' THEN '军人'
        WHEN 'Y' THEN '不便分类的其他从业人员'
        WHEN 'Z' THEN '未知'
        ELSE '未知'
    END AS position_ppl --职业
-- , --是否授信

-- ,--用户可用额度
-- ,--用户授信额度
    ,SUBSTRING(id_no, 1, 6) as id_card_addr --身份证归属地
-- ,--当前逾期天数
-- ,--历史逾期天数
FROM dwd_user_info_real_source
WHERE create_time is not NULL
    and hg_binlog_event_type in ('5','7')
    and sex_ocr is not NULL
    and mobile_no is not NULL
;

CREATE TEMPORARY VIEW user_info_view
AS SELECT
    basic_mobile
    ,YEAR(CURRENT_DATE) - YEAR(birth_date) - CASE
        WHEN MONTH(CURRENT_DATE) < MONTH(birth_date) THEN 1
        WHEN MONTH(CURRENT_DATE) = MONTH(birth_date)
        AND DAYOFMONTH(CURRENT_DATE) < DAYOFMONTH(birth_date) THEN 1
        ELSE 0
    END AS age_ppl
    ,*
    ,area.region_name as livingCity_ppl
    ,area2.region_name as idCity_ppl
    -- ,s.business_line
    -- ,s.mobile
    ,CASE
        WHEN s.status = 'S' THEN '是'
        WHEN s.status = 'F' THEN '否'
        WHEN s.status = 'R' THEN '否'
        ELSE '-'
    END as if_credit_ppl
FROM user_info
    LEFT join zl_area_code_source FOR SYSTEM_TIME AS OF PROCTIME() area
        on user_info.city_code = area.region_code
    LEFT JOIN zl_area_code_source FOR SYSTEM_TIME AS OF PROCTIME() area2
        ON user_info.id_card_addr = area2.region_code
    LEFT JOIN LATERAL (
    select
        sub.customer_no
        ,sub.status
    from (
        select
            z.customer_no
            ,z.status
            ,row_number()
            OVER (
                partition by z.customer_no
                ORDER BY  z.id DESC
            ) rk
        from zx_credit_applicant_result_source z
        where z.customer_no = user_info.customer_no
            and z.status <> 'P'
    ) sub
    where rk = 1
) s
        ON TRUE
;
;

CREATE TEMPORARY TABLE if not EXISTS zl_area_code_source
(
    region_code        VARCHAR
    ,region_short_name VARCHAR
    ,region_name       VARCHAR

    -- ,WATERMARK FOR hg_binlog_timestamp_us AS hg_binlog_timestamp_us - INTERVAL '20' SECOND  -- 允许最大5秒的数据延迟
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zl_area_code'
    ,'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

-- 用户额度

CREATE TEMPORARY VIEW credit_amt_view
AS SELECT
    CAST(CAST(ROUND(
        CAST(amt_info.credit_amount AS NUMERIC)
        ,2
    ) as DECIMAL(10, 2)) AS VARCHAR) as credit_limit_ppl   -- APP产品授信额度
    -- ,amt_info.used_amount   -- App产品已用额度
    ,CAST(cast(ROUND(
        CAST(amt_info.used_amount AS NUMERIC)
        ,2
    ) as DECIMAL(10, 2)) AS VARCHAR) as used_amount   -- App产品已用额度
    -- ,(amt_info.credit_amount - amt_info.used_amount) as available_limit_ppl -- 产品域可用额度
    ,CAST(CAST(ROUND(
        CAST(amt_info.credit_amount AS NUMERIC) - CAST(amt_info.used_amount AS NUMERIC)
        ,2
    ) AS DECIMAL(10, 2)) AS VARCHAR) AS available_limit_ppl
    ,CAST(ROUND(
        CAST(amt_info.credit_amount AS NUMERIC) - CAST(amt_info.used_amount AS NUMERIC)
        ,2
    ) AS VARCHAR) AS available_limit_ppl_str
    ,CASE amt_info.status
        WHEN 'NR' THEN '正常'
        WHEN 'TF' THEN '冻结'
        ELSE '-'
    END as credi_status_ppl
    ,user_info.basic_mobile as mobile_no
    ,user_info.user_source_var as user_source_var
FROM product_credit_amt_detail_source2 as amt_info
    LEFT JOIN user_info
        ON amt_info.customer_no = user_info.customer_no
        and amt_info.product_code = user_info.product_code
WHERE hg_binlog_event_type in ('5','7')
;

-- select * from credit_amt_view where mobile_no='18768406195';
-- select * from product_credit_amt_detail_source2;
-- select * from credit_amt_view;



-----是否授信



-----逾期天数

CREATE TEMPORARY TABLE if not EXISTS zx_loan_plan_info_source
(
    id                        bigint not null
    ,plan_no                  VARCHAR
    ,compensate_sign          VARCHAR
    ,compensate_date          timestamp
    ,overdue_day              bigint
    ,term                     bigint
    ,start_date               VARCHAR
    ,due_date                 VARCHAR
    ,plan_status              VARCHAR
    ,repay_time               timestamp
    ,prin_amt                 VARCHAR
    ,int_amt                  VARCHAR
    ,oint_amt                 VARCHAR
    ,fee_amt                  VARCHAR
    ,late_fee_amt             VARCHAR
    ,other_amt                VARCHAR
    ,guarantee_amt            VARCHAR
    ,advice_amt               VARCHAR
    ,insure_amt               VARCHAR
    ,deduct_amt               VARCHAR
    ,act_prin_amt             VARCHAR
    ,act_int_amt              VARCHAR
    ,act_oint_amt             VARCHAR
    ,act_fee_amt              VARCHAR
    ,act_late_fee_amt         VARCHAR
    ,act_other_amt            VARCHAR
    ,act_guarantee_amt        VARCHAR
    ,act_advice_amt           VARCHAR
    ,act_insure_amt           VARCHAR
    ,act_deduct_amt           VARCHAR
    ,grace_date               VARCHAR
    ,settle_time              VARCHAR
    ,repayment_no             VARCHAR
    ,creation_time            timestamp
    ,del_flag                 VARCHAR
    ,applicant_id             bigint
    ,type                     VARCHAR
    ,update_time              timestamp(3)
    ,channel_code             VARCHAR
    ,funds_code               VARCHAR
    ,compensate_status        bigint
    ,deduct_oint_amt          VARCHAR
    ,deduct_guarantee_amt     VARCHAR
    ,deduct_advice_amt        VARCHAR
    ,act_deduct_oint_amt      VARCHAR
    ,act_deduct_guarantee_amt VARCHAR
    ,act_deduct_advice_amt    VARCHAR
    ,deduct_prin_amt          VARCHAR
    ,deduct_int_amt           VARCHAR
    ,act_deduct_prin_amt      VARCHAR
    ,act_deduct_int_amt       VARCHAR
    ,PRIMARY KEY (id) NOT ENFORCED
        -- ,hg_binlog_lsn          BIGINT
    ,hg_binlog_event_type     BIGINT
    -- ,hg_binlog_timestamp_us BIGINT
    -- ,event_time timestamp(3)
    -- ,WATERMARK FOR create_time AS create_time - INTERVAL '10' SECOND  -- 允许最大5秒的数据延迟
    ,proc_time                AS PROCTIME()
)
WITH (
    'connector' = 'hologres'
    -- ,'dbname' = 'data_dws'
    ,'dbname' = '${secret_values.holo_dbname_data_dws_2}'
    ,'tablename' = 'zws_middleware.zx_loan_plan_info'
    ,'binlog' = 'true'
    -- ,'upsertSource' = 'true'
    -- ,'binlogStartupMode' = 'initial'
    ,'sdkMode' = 'jdbc'
    ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

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
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zl_drms_judgements'
    -- ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    -- ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

CREATE TEMPORARY TABLE if not EXISTS zx_loan_apply_record_source
(
    id             bigint NOT NULL
    ,loan_apply_no VARCHAR
    ,apply_id      VARCHAR
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'zws_middleware.zx_loan_apply_record'
    -- ,'binlog' = 'true'
    ,'sdkMode' = 'jdbc'
    -- ,'cdcmode' = 'true'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;

CREATE TEMPORARY VIEW zy_loan_plan
AS SELECT
    plan.plan_no
    ,SPLIT_INDEX(plan.plan_no, '_', 0) as loan_apply_no
    ,'' as loan_no
    ,plan.term
    ,plan.overdue_day
    ,plan.plan_status
    ,plan.channel_code
    ,plan.funds_code
    ,(
        CAST(plan.prin_amt AS DECIMAL(12, 2)) - CAST(plan.act_prin_amt AS DECIMAL(12, 2))) as ps_rem_prcp
    ,plan.proc_time
    ,'zws_middleware' AS data_source
FROM zx_loan_plan_info_source plan
where del_flag = '0'
    and hg_binlog_event_type in ('5','7')
;

CREATE TEMPORARY VIEW loan_plan_group_view_1
AS SELECT
    loan_apply_no
    ,MAX(overdue_day) as loan_max_overdue_day
from zy_loan_plan
where loan_apply_no is not NULL
    and plan_status = '1'
GROUP BY
    TUMBLE(proc_time, INTERVAL '2' SECOND)
    ,loan_apply_no
;

CREATE TEMPORARY VIEW loan_plan_group_view_2
AS SELECT
    loan_apply_no
    ,MAX(overdue_day) as loan_max_his_overdue_day
    -- ,MAX(IF(ps_rem_prcp > 0, overdue_day, -1)) as loan_max_overdue_day
    -- ,CASE WHEN plan_status='2' THEN -1
    -- WHEN plan_status in ('0','1') THEN MAX(overdue_day)
    -- END AS loan_max_overdue_day
    ,sum(IF(ps_rem_prcp > 0, 1, 0)) as setl_flag
    ,sum(ps_rem_prcp) as loan_credit_amount_used
from zy_loan_plan
where loan_apply_no is not NULL
-- GROUP BY loan_apply_no
GROUP BY
    TUMBLE(proc_time, INTERVAL '2' SECOND)
    ,loan_apply_no
;

CREATE TEMPORARY VIEW loan_plan_group_view
AS SELECT
    t2.*
    ,t1.loan_max_overdue_day
from loan_plan_group_view_2 as t2
    LEFT JOIN loan_plan_group_view_1 as t1
        ON t2.loan_apply_no = t1.loan_apply_no
where t2.loan_max_his_overdue_day > 0
;


-- select * from loan_plan_group_view where loan_max_his_overdue_day<>loan_max_overdue_day;
-- select * from loan_plan_group_view where loan_apply_no='LR6467094108697534464';
-- select * from loan_plan_group_view;

CREATE TEMPORARY VIEW max_overdue_view
as SELECT
    t1.loan_apply_no
    ,t1.loan_max_his_overdue_day
    ,t1.loan_max_overdue_day
    ,t3.user_idcard_number
    ,t3.user_mobile_number as mobile_no
    ,t3.channel_code
    ,CASE
        WHEN t3.channel_code IN ('APPZY', 'ICE_ZLSK_36', 'ZL_HALUO', 'HL_RL', 'HY', 'JD_RL', 'QXL', 'RS', 'RS_RL',
                              'VIVO_RL', 'XL', 'XL_LY', 'YQG', 'ZL_GM', 'ZL_HSR', 'ZL_HY', 'ZL_WC') THEN '众利'
        WHEN t3.channel_code IN ('LLH_HSR', 'LLH_R360', 'LLH_RP', 'LLH_XY') THEN '龙力花'
        WHEN t3.channel_code IN ('LXJ', 'R360', 'RP', 'TYH', 'TYH_APPZY', 'TYH_HSR', 'TYH_HY', 'TYH_JKQB', 'TYH_JQB',
                              'TYH_JXC', 'TYH_KN', 'TYH_LXJ', 'TYH_LXJPLUS', 'TYH_R360', 'TYH_RP', 'TYH_RPPLUS',
                              'TYH_SD', 'TYH_XD', 'TYH_XY', 'ZZJT') THEN '天源花'
    END AS user_source_var
FROM loan_plan_group_view as t1
    LEFT JOIN zx_loan_apply_record_source FOR SYSTEM_TIME AS OF PROCTIME() as t2
        ON t1.loan_apply_no = t2.loan_apply_no
    LEFT JOIN zy_zl_drms_judgements_source FOR SYSTEM_TIME AS OF PROCTIME() as t3
        ON t2.apply_id = t3.apply_id
;

CREATE TEMPORARY TABLE gio_user_message_sink
(
    event_time                BIGINT NOT NULL
    ,-- 事件时间
    anonymous_id             varchar
    ,-- 访问用户 ID
    login_user_key           varchar
    ,-- 登录用户 KEY
    login_user_id            varchar
    ,-- 登录用户 ID
    basic_mobile             varchar NOT NULL
    ,-- 用户手机号
    last_name_ppl            varchar
    ,-- 姓
    age_ppl                  varchar
    ,-- 年龄
    gender_ppl               varchar
    ,-- 性别
    marriage_ppl             varchar
    ,-- 婚姻
    education_ppl            varchar
    ,-- 学历
    living_city_ppl          varchar
    ,-- 居住地_市
    income_ppl               varchar
    ,-- 月收入
    industry_ppl             varchar
    ,-- 行业
    position_ppl             varchar
    ,-- 职业
    if_credit_ppl            varchar
    ,-- 是否授信
    credi_status_ppl         varchar
    ,-- 额度状态
    available_limit_ppl      varchar
    ,-- 用户可用额度
    credit_limit_ppl         varchar
    ,-- 用户授信额度
    id_city_ppl              varchar
    ,-- 身份证归属地
    overdue_days_ppl         varchar
    ,-- 当前逾期天数
    history_overdue_days_ppl varchar
    ,-- 历史逾期天数
    channel_key              varchar
    ,-- code
    user_source_var          varchar
    ,-- 业务线
    business_line            varchar
    ,-- 业务线，可多个
    if_black_list            varchar
    ,report_status            varchar -- 上报状态 0,1
)
WITH (
    'connector' = 'hologres'
    ,'dbname' = '${secret_values.holo_dbname_data_dws}'
    ,'tablename' = 'realtime.gio_user_message_prd'
    ,-- 'mutatetype' = 'insertorreplace',
    -- 'ignoredelete' = 'true',
    -- 'sink.delete-strategy' = 'DELETE_ROW_ON_PK',
    -- 'partial-insert.enabled' = 'true',
    -- 'binlog' = 'true',
    'sdkMode' = 'jdbc'
    ,'endpoint' = '${secret_values.holocatalog_endpoint}'
    ,'username' = '${secret_values.holocatalog_username}'
    ,'password' = '${secret_values.holocatalog_password}'
)
;



-- select * from user_info_view;
-- select * from result_table;


-- select * from credit_amt_view;
-- select * from product_credit_amt_detail_source2;

BEGIN STATEMENT SET
;


-- , event_time, anonymous_id, login_user_key, login_user_id, basic_mobile, last_name_ppl, age_ppl, gender_ppl, marriage_ppl, education_ppl, living_city_ppl, income_ppl, industry_ppl, position_ppl, if_credit_ppl, credi_status_ppl, available_limit_ppl, credit_limit_ppl, id_city_ppl, overdue_days_ppl, history_overdue_days_ppl, report_status

INSERT INTO gio_user_message_sink (
    event_time
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,basic_mobile
    ,last_name_ppl
    ,age_ppl
    ,gender_ppl
    ,marriage_ppl
    ,education_ppl
    ,living_city_ppl
    ,income_ppl
    ,industry_ppl
    ,position_ppl
    ,if_credit_ppl
    -- ,credi_status_ppl
    -- ,available_limit_ppl
    -- ,credit_limit_ppl
    ,id_city_ppl
    -- ,overdue_days_ppl
    -- ,history_overdue_days_ppl
    ,channel_key
    ,user_source_var
    ,business_line
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'' as anonymous_id
    ,'$basic_userId' as login_user_key
    ,basic_mobile as login_user_id
    ,basic_mobile as basic_mobile
    ,lastName_ppl as last_name_ppl
    ,CAST(age_ppl as VARCHAR) as age_ppl
    ,gender_ppl
    ,marriage_ppl
    ,education_ppl
    ,livingCity_ppl as living_city_ppl
    ,income_ppl
    ,industry_ppl
    ,position_ppl
    ,if_credit_ppl
    -- ,credi_status_ppl
    -- ,CAST(available_limit_ppl as STRING) as available_limit_ppl
    -- ,CAST(credit_limit_ppl as STRING) as credit_limit_ppl
    ,idCity_ppl as id_city_ppl
    ,channel_key
    ,user_source_var
    ,user_source_var as business_line
    ,GrowingUserF_PRD(
        user_source_var
        ,CAST(basic_mobile as STRING)
        ,MAP['basic_mobile' ,  basic_mobile
        ,'lastName_ppl' , lastName_ppl
        ,'age_ppl' , CAST(age_ppl as VARCHAR)
        ,'gender_ppl' , gender_ppl
        ,'marriage_ppl', marriage_ppl
        ,'education_ppl' , education_ppl
        ,'livingCity_ppl' , livingCity_ppl
        ,'income_ppl' , income_ppl
        ,'industry_ppl' , industry_ppl
        ,'position_ppl' , position_ppl
        ,'idCity_ppl' , idCity_ppl
        ,'ifBlacklist' , ''
        ,'businessLine' , user_source_var
        ,'ifCredit_ppl',if_credit_ppl]
    ) as report_status
FROM user_info_view
WHERE basic_mobile is not NULL
;


-- -----额度变化

INSERT INTO gio_user_message_sink (
    event_time
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,basic_mobile
    ,credi_status_ppl
    ,available_limit_ppl
    ,credit_limit_ppl
    ,user_source_var
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'' as anonymous_id
    ,'$basic_userId' as login_user_key
    ,mobile_no as login_user_id
    ,mobile_no as basic_mobile
    -- , as ifCredit_ppl
    ,credi_status_ppl
    ,available_limit_ppl
    ,credit_limit_ppl
    ,user_source_var
    ,GrowingUserF_PRD(
        user_source_var
        ,CAST(mobile_no as STRING)
        ,MAP['basic_mobile' ,  mobile_no
        ,'crediStatus_ppl' , credi_status_ppl
        ,'availableLimit_ppl' , available_limit_ppl
        ,'creditLimit_ppl' , credit_limit_ppl]
    ) as report_status
FROM credit_amt_view
WHERE mobile_no is not NULL
;


--- 逾期天数

INSERT INTO gio_user_message_sink (
    event_time
    ,anonymous_id
    ,login_user_key
    ,login_user_id
    ,basic_mobile
    ,overdue_days_ppl
    ,history_overdue_days_ppl
    ,user_source_var
    ,report_status
)
SELECT
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) * 1000 as event_time
    ,'' as anonymous_id
    ,'$basic_userId' as login_user_key
    ,mobile_no as login_user_id
    ,mobile_no as basic_mobile
    ,CAST(loan_max_overdue_day as VARCHAR) as overdue_days_ppl
    ,CAST(loan_max_his_overdue_day as VARCHAR)  as history_overdue_days_ppl
    ,user_source_var
    ,GrowingUserF_PRD(
        user_source_var
        ,CAST(mobile_no as STRING)
        ,MAP['basic_mobile' ,  mobile_no
        ,'overdueDays_ppl' , CAST(loan_max_overdue_day as VARCHAR)
        ,'historyOverdueDays_ppl' , CAST(loan_max_his_overdue_day as VARCHAR)]
    ) as report_status
FROM max_overdue_view
WHERE mobile_no is not NULL
;

END
;







-- SELECT
--      mobile
--     ,GrowingUser(CAST(mobile as STRING )
--         , MAP['key1', 'value1',
--             'key2', 'value2' ]

--         )

-- FROM zx_credit_user_info_source;




