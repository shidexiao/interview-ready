create table if not exists vdm_ds_dev.fhj_model_score_bin_analysis_v14 as
select a.order_id 
	  ,a.loan_type
	  -- ,a.product_code
	  ,substr(apply_time, 1, 7) as apply_month
	  ,t.plat
	  ,t.is_loan
	  ,t.is_eff_s1d15
	  ,t.is_eff_s3d15
	  ,(case when t.is_eff_s1d15 = 1 then t.s1d15 else -1 end) as s1d15
	  ,(case when t.is_eff_s3d15 = 1 then t.s3d15 else -1 end) as s3d15

	  ,a.finalscore  
	  ,case when cast(finalscore   as double) <= percent[0] then concat('0.000000', ' - ', substr(cast(percent[0] as string), 1, 8))
	        when cast(finalscore   as double) > percent[0] and cast(finalscore   as double) <= percent[1] then concat(substr(cast(percent[0] as string), 1, 8), ' - ', substr(cast(percent[1] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[1] and cast(finalscore   as double) <= percent[2] then concat(substr(cast(percent[1] as string), 1, 8), ' - ', substr(cast(percent[2] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[2] and cast(finalscore   as double) <= percent[3] then concat(substr(cast(percent[2] as string), 1, 8), ' - ', substr(cast(percent[3] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[3] and cast(finalscore   as double) <= percent[4] then concat(substr(cast(percent[3] as string), 1, 8), ' - ', substr(cast(percent[4] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[4] and cast(finalscore   as double) <= percent[5] then concat(substr(cast(percent[4] as string), 1, 8), ' - ', substr(cast(percent[5] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[5] and cast(finalscore   as double) <= percent[6] then concat(substr(cast(percent[5] as string), 1, 8), ' - ', substr(cast(percent[6] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[6] and cast(finalscore   as double) <= percent[7] then concat(substr(cast(percent[6] as string), 1, 8), ' - ', substr(cast(percent[7] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[7] and cast(finalscore   as double) <= percent[8] then concat(substr(cast(percent[7] as string), 1, 8), ' - ', substr(cast(percent[8] as string), 1, 8))
	  	    when cast(finalscore   as double) > percent[8] then concat(substr(cast(percent[8] as string), 1, 8), ' - ', '+inf')
	        else null
	  end as score_range

-- from dm_ds_fraud.gyd_pd_submodel_score_feature_v1 a
-- from dm_ds_fraud.fhj_rpd_syj_acard_v2_submodel_score_feature a
   from dm_ds_fraud.gyd_rpd_acard_scorecard_v3 a
join 
	(
		-- select loan_type, product_code,
		select loan_type,
			   percentile_approx(cast(finalscore   as double), array(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), 9999) as percent
		from dm_ds_fraud.gyd_rpd_acard_scorecard_v3
		where dt >= '2018-01-01' and dt <= '2019-06-01'
	     -- and loan_type = 20
		group by loan_type
		-- , product_code                      -- 在有效订单层进行分箱，这样便于后面的比较
	) p
on a.loan_type = p.loan_type
-- and a.product_code = p.product_code

join dm_ds_fraud.loanpro_order_overdue_status_t t
on a.order_id = t.order_original_id

-- where a.loan_type = 20