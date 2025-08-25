	SELECT  
		user_id
	   ,SUM(if(overdue_day > 0 AND step_no = 1 AND datediff(date_sub(current_date(), 2),to_date(due_day)) +1 > 30 ,due_amount,0)) AS dpd1_at_mob1_due_amount
       ,SUM(if(overdue_day > 30 AND step_no = 1 ,due_amount,0)) AS dpd30_at_mob1_due_amount
	   ,SUM(if(overdue_day > 0 AND step_no = 3 AND datediff(date_sub(current_date(), 2),to_date(due_day)) +1 > 30 AND due_day_curr_ovd_status <> '1',due_amount,0)) AS dpd1_at_mob3_due_amount
       ,SUM(if(overdue_day > 30 AND step_no = 3 AND  due_day_curr_ovd_status <> '1',due_amount,0)) AS dpd30_at_mob3_due_amount
	FROM edw_tinyv.e_repay_plan_core_d a
	WHERE dt = date_sub(current_date(), 2)
	AND substr(loan_time,1,10) >= '2021-08-01'
	AND reg_brand_name IN ('weibo_jieqian')
	AND sku IN ('cycle_loan','flexible_loan')
	GROUP BY user_id