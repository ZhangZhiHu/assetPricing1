# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-07  15:01
# NAME:assetPricing1-template.py
import os
import pandas as pd
import numpy as np
from zht.utils.dfu import get_first_group, get_group_n
from zht.utils.mathu import get_outer_frame

from dataset import *
from dout import read_df
from config import *

from zht.utils import assetPricing
from zht.utils.assetPricing import summary_statistics, cal_breakPoints, count_groups, famaMacBeth

class CrossSection:
    def __init__(self):
        self.mktRetM=read_df('mktRetM','M')
        self.ff3=read_df('ff3','M')

    def _excess_ret(self,series):
        # series.name = series.name.replace('-', '_')  # '-' should not appear in formula
        lags = 5  # TODO: setting for lags

        #TODO:dropna?
        df=pd.DataFrame(series)
        df.columns = ['y']
        df=df.dropna()

        formula = 'y ~ 1'
        nw = assetPricing.newey_west(formula, df, lags)
        return nw['Intercept'].rename(index={'coef': 'excess return',
                                             't': 'excess return t'})

    def _alpha(self,series):
        '''
        series can also be a mutiIndex pd.Series

        :param series:
        :return:
        '''
        lags = 5  # TODO: setting for lags
        df=pd.DataFrame(series)
        df.columns=['y']
        df=df.join(self.mktRetM)
        df=df.dropna()

        # TODO:the index in series represent time t,and its e_ahead represennt the returns in time t+1，
        # TODO:we need to shift the index of x here to match the data.Why don't we adjust the index in dataset to make
        # TODO:the index represent time t+1(or t) and all the indicators are calculated in time t(or t-1).
        formula='y ~ mktRetM'
        nw = assetPricing.newey_west(formula, df, lags=lags)
        return nw['Intercept'].rename(index={'coef': 'alpha',
                                             't': 'alpha_t'})

    def _ff3_alpha(self,series):
        lags=5
        df=pd.DataFrame(series)
        df.columns=['y']
        df=df.join(self.ff3.copy())
        df=df.dropna()

        formula='y ~ rp + smb + hml'
        nw=assetPricing.newey_west(formula,df,lags)
        return nw['Intercept'].rename(index={'coef': 'ff3_alpha',
                                             't': 'ff3_alpha_t'})


class Univariate(CrossSection):
    '''
    TODO:add a tool to analyse the relationship between different characteristics,
    as table II of Pan, Li, Ya Tang, and Jianguo Xu. “Speculative Trading and Stock Returns.” Review of Finance 20, no. 5 (August 2016): 1835–65. https://doi.org/10.1093/rof/rfv059.

    '''
    q=10

    def __init__(self,factor,path):
        super().__init__()
        self.factor=factor
        self.path=path
        self.name=factor.name
        self.indicators=factor.indicators
        self.df=dataset.get_by_factorname(factor.name)
        self.groupnames=[self.name+str(i) for i in range(1,self.q+1)]

    def summary(self):
        df=dataset.get_by_factorname(self.name)
        series=[]
        for indicator in self.indicators:
            s=summary_statistics(df[indicator].unstack())
            series.append(s.mean())
        pd.concat(series,keys=self.indicators,axis=1).to_csv(os.path.join(self.path,'summary.csv'))

    def correlation(self, indicators=None):
        if not indicators:
            indicators=self.indicators

        comb=dataset.get_by_indicators(indicators)

        def _spearman(df):
            df=df.dropna()
            if df.shape[0]>10:#TODO:thresh to choose
                return assetPricing.corr(df,'spearman',winsorize=False)

        def _pearson(df):
            df=df.dropna()
            if df.shape[0]>10:
                return assetPricing.corr(df,'pearson',winsorize=True)

        corrs=comb.groupby('t').apply(_spearman)
        corrp=comb.groupby('t').apply(_pearson)

        corrsAvg=corrs.groupby(level=1).mean().reindex(index=indicators, columns=indicators)
        corrpAvg=corrp.groupby(level=1).mean().reindex(index=indicators, columns=indicators)

        corr1 = np.tril(corrpAvg.values, k=-1)
        corr2 = np.triu(corrsAvg.values, k=1)

        corr = pd.DataFrame(corr1 + corr2, index=corrpAvg.index, columns=corrpAvg.columns)
        np.fill_diagonal(corr.values, np.NaN)
        corr.to_csv(os.path.join(self.path, 'corr.csv'))
        # corrpAvg.to_csv(os.path.join(self.path, 'corr_pearson.csv'))
        # corrsAvg.to_csv(os.path.join(self.path, 'corr_spearman.csv'))

    def persistence(self):
        perdf=pd.DataFrame()
        for indicator in self.indicators:
            per=assetPricing.persistence(self.df[indicator].unstack(),
                                         offsets=[1, 3, 6, 12, 24, 36, 48, 60, 120])
            perdf[indicator]=per
        perdf.to_csv(os.path.join(self.path,'persistence.csv'))

    def breakPoints_and_countGroups(self):
        for indicator in self.indicators:
            d=self.df[indicator].unstack()
            d=d.dropna(axis=0,how='all')#there is no samples for some months due to festival
            bps=cal_breakPoints(d,self.q)
            bps.to_csv(os.path.join(self.path,'breakPoints_%s.csv'%indicator))
            count=count_groups(d,self.q)
            count.to_csv(os.path.join(self.path,'count_%s.csv'%indicator))
        #TODO: put them in one csv (stacked)

        # TODO:In fact,the count is not exactly the number of stocks to calculate the weighted return
        # TODO:as some stocks will be deleted due to the missing of weights.

    def _get_port_data(self,indicator):
        groupid=dataset.get_by_indicators([indicator])
        groupid['g']=groupid.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[indicator],self.q,
                              labels=[indicator+str(i) for i in range(1,self.q+1)])
        )
        return groupid

    def portfolio_characteristics(self,sortedIndicator,otherIndicators):
        '''
        as table 12.3 panel A
        :param sortedIndicator:
        :param otherIndicators:
        :return:
        '''
        groupid=self._get_port_data(sortedIndicator)
        comb=dataset.get_by_indicators(otherIndicators)
        comb=pd.concat([groupid,comb],axis=1)
        characteristics_avg=comb.groupby(['t','g']).mean().groupby('g').mean()
        characteristics_avg.to_csv(os.path.join(self.path,'portfolio characteristics.csv'))

    def portfolio_analysis(self):
        all_indicators=list(set(self.indicators+['mktCap','eret','mktRetM']))
        comb=dataset.get_by_indicators(all_indicators)
        #
        # if self.name=='size':
        #     comb=dataset.get_by_indicators(self.indicators+['eret','mktRetM'])
        # else:
        #     comb=dataset.get_by_indicators(self.indicators+['mktCap','eret','mktRetM'])

        comb=comb.dropna()

        def _get_bc(group_ts):
            unstk = group_ts.unstack()
            unstk.columns = unstk.columns.astype(str)
            unstk.index = unstk.index.astype(str)
            # TODO: apply this method in Bivariate
            unstk['_'.join([self.groupnames[-1], self.groupnames[0]])] = unstk[self.groupnames[-1]] - unstk[
                self.groupnames[0]]
            stk = unstk.stack()
            stk.name = 'eret'
            # TODO: group_wavg_ts
            # table 8.4
            # part B
            _b = stk.groupby(gcol).apply(self._excess_ret)
            # part C
            _c = stk.groupby(gcol).apply(self._alpha)

            table = pd.concat([_b.unstack(), _c.unstack()], axis=1).T
            return table

        tables_e=[]
        tables_w=[]
        for indicator in self.indicators:
            gcol='g_%s'%indicator
            comb[gcol]=comb.groupby('t',group_keys=False).apply(
                lambda df:pd.qcut(df[indicator],self.q,labels=self.groupnames))
            group_eavg_ts=comb.groupby(['t',gcol])['eret'].mean()

            if self.name=='size':
                # It should be noted that when MktCap is used as the sort variable,it is also used as the measure of
                # market capitalization when weighting the portfolios.When mktCap_ff is used as the sort variable,
                # we also use mktCap_ff to weight the portfolios.As page 159.
                group_wavg_ts=comb.groupby(['t',gcol]).apply(
                    lambda df:np.average(df['eret'],weights=df[indicator])
                )
            else:
                group_wavg_ts=comb.groupby(['t',gcol]).apply(
                    lambda df:np.average(df['eret'],weights=df['mktCap'])
                #TODO:how about the nan
                )

            #table 8.4 part A
            a_data=comb.groupby(['t',gcol])[indicator].mean()
            a_data=a_data.unstack()
            a_data.columns=a_data.columns.astype(str)
            a_data.index=a_data.index.astype(str)
            a_data['_'.join([self.groupnames[-1],self.groupnames[0]])]=a_data[self.groupnames[-1]]-a_data[self.groupnames[0]]
            _a=a_data.mean()

            #equal weighted
            ab_e=_get_bc(group_eavg_ts)
            _a.name=indicator
            table_e=pd.concat([_a.to_frame().T,ab_e],axis=0)

            #value weighted,it only have part B and part C
            table_w=_get_bc(group_wavg_ts)

            tables_e.append(table_e)
            tables_w.append(table_w)

            print(indicator)

        pd.concat(tables_e,axis=0,keys=self.indicators).to_csv(
            os.path.join(os.path.join(self.path,'univariate portfolio analysis-equal weighted.csv')))
        pd.concat(tables_w,axis=0,keys=self.indicators).to_csv(
            os.path.join(os.path.join(self.path,'univariate portfolio analysis-value weighted.csv')))

    def portfolio_analysis_with_ff3_alpha(self):
        all_indicators=list(set(self.indicators+['mktCap','eret','mktRetM']))
        comb=dataset.get_by_indicators(all_indicators)
        comb=comb.dropna()

        def _get_bc(group_ts,gcol):
            unstk = group_ts.unstack()
            unstk.columns = unstk.columns.astype(str)
            unstk.index = unstk.index.astype(str)
            # TODO: apply this method in Bivariate
            unstk['_'.join([self.groupnames[-1], self.groupnames[0]])] = unstk[self.groupnames[-1]] - unstk[
                self.groupnames[0]]
            stk = unstk.stack()
            stk.name = 'eret'
            # TODO: group_wavg_ts
            # table 8.4
            # part B
            _b = stk.groupby(gcol).apply(self._excess_ret)
            # part C
            _c = stk.groupby(gcol).apply(self._alpha)

            ########################################################################
            #only this two rows is different with self.portfolio_analysis()
            # part D
            _d=stk.groupby(gcol).apply(self._ff3_alpha)

            table = pd.concat([_b.unstack(), _c.unstack(),_d.unstack()], axis=1).T
            #########################################################################
            return table

        tables_e = []
        tables_w = []
        for indicator in self.indicators:
            gcol = 'g_%s' % indicator
            comb[gcol] = comb.groupby('t', group_keys=False).apply(
                lambda df: pd.qcut(df[indicator], self.q, labels=self.groupnames))
            group_eavg_ts = comb.groupby(['t', gcol])['eret'].mean()

            if self.name == 'size':
                # It should be noted that when MktCap is used as the sort variable,it is also used as the measure of
                # market capitalization when weighting the portfolios.When mktCap_ff is used as the sort variable,
                # we also use mktCap_ff to weight the portfolios.As page 159.
                group_wavg_ts = comb.groupby(['t', gcol]).apply(
                    lambda df: np.average(df['eret'], weights=df[indicator])
                )
            else:
                group_wavg_ts = comb.groupby(['t', gcol]).apply(
                    lambda df: np.average(df['eret'], weights=df['mktCap'])
                    # TODO:how about the nan
                )

            # table 8.4 part A
            a_data = comb.groupby(['t', gcol])[indicator].mean()
            a_data = a_data.unstack()
            a_data.columns = a_data.columns.astype(str)
            a_data.index = a_data.index.astype(str)
            a_data['_'.join([self.groupnames[-1], self.groupnames[0]])] = a_data[self.groupnames[-1]] - a_data[
                self.groupnames[0]]
            _a = a_data.mean()

            # equal weighted
            ab_e = _get_bc(group_eavg_ts,gcol)
            _a.name = indicator
            table_e = pd.concat([_a.to_frame().T, ab_e], axis=0)

            # value weighted,it only have part B and part C
            table_w = _get_bc(group_wavg_ts,gcol)

            tables_e.append(table_e)
            tables_w.append(table_w)

            print(indicator)

    def fm(self):
        comb=dataset.get_by_indicators(self.indicators+['eret'])
        data=[]
        for indicator in self.indicators:
            subdf=comb[[indicator,'eret']]
            subdf=subdf.dropna()
            subdf.columns=['y','x']
            subdf=subdf.reset_index()
            formula='y ~ x'
            r,adj_r2,n=famaMacBeth(formula,'t',subdf,lags=5)
            data.append([r.loc['x', 'coef'], r.loc['x', 'tvalue'],
                         r.loc['Intercept', 'coef'], r.loc['Intercept', 'tvalue'],
                         adj_r2, n])

        result = pd.DataFrame(data, index=self.indicators,
                              columns=['slope', 't', 'Intercept', 'Intercept_t', 'adj_r2', 'n']).T
        result.to_csv(os.path.join(self.path, 'fama macbeth regression analysis.csv'))

    def run(self):
        self.summary()
        self.correlation()
        self.persistence()
        self.breakPoints_and_countGroups()
        self.portfolio_analysis()
        self.fm()


class Beta(Univariate):
    def __init__(self):
        super().__init__(BETA,BETA_PATH)


class Size(Univariate):
    def __init__(self):
        super().__init__(SIZE,SIZE_PATH)

    def get_percent_ratio(self):

        def _get_ratio(s):
            ratios = [1, 5, 10, 25]
            return pd.Series([s.nlargest(r).sum() / s.sum() for r in ratios],
                             index=ratios)

        df=dataset.get_by_indicators('mktCap')
        d=df.groupby('t').apply(_get_ratio)
        fig=d.unstack().plot().get_figure()
        fig.savefig(os.path.join(self.path,'percent of market value.png'))

    def run(self):
        super().run()
        self.get_percent_ratio()


class Value(Univariate):
    def __init__(self):
        super().__init__(VALUE,VALUE_PATH)

    def correlation(self,indicators=('bm','lgbm','12M','size')):
        super().correlation(indicators)


class Univariate_mom(Univariate):
    def __init__(self):
        super().__init__(MOM,MOM_PATH)

    def correlation(self,indicators=MOM_NAMES+['12M','size','bm']):
        super().correlation(indicators)

    def _one_indicator(self,indicator):
        ns=range(0,12)
        all_indicators=list(set([indicator]+['mktCap','eret','mktRetM']))
        comb=dataset.get_by_indicators(all_indicators)
        comb=comb.dropna()
        comb['g']=comb.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[indicator],self.q,
                              labels=[indicator+str(i) for i in range(1,self.q+1)])
        )


        def _one_indicator_one_weight_type(group_ts, indicator):
            def _big_minus_small(s,indicator):
                time=s.index.get_level_values('t')[0]
                return s[(time,indicator+str(self.q))]-s[(time,indicator+'1')]

            spread_data=group_ts.groupby('t').apply(lambda s:_big_minus_small(s,indicator))
            _a=self._excess_ret(spread_data)
            _b=self._alpha(spread_data)
            _c=self._ff3_alpha(spread_data)
            s=pd.concat([_a,_b,_c],axis=0)
            return s

        eret=comb['eret'].unstack()

        s_es=[]
        s_ws=[]
        eret_names=[]
        for n in ns:
            eret_name='eret_ahead%s'%(n+1)
            comb[eret_name]=eret.shift(-n).stack()

            group_eavg_ts=comb.groupby(['t','g'])[eret_name].mean()
            group_wavg_ts=comb.groupby(['t','g']).apply(lambda df:np.average(df[eret_name],weights=df['mktCap']))

            s_e=_one_indicator_one_weight_type(group_eavg_ts,indicator)
            s_w=_one_indicator_one_weight_type(group_wavg_ts,indicator)
            s_es.append(s_e)
            s_ws.append(s_w)
            eret_names.append(eret_name)
        eq_table=pd.concat(s_es,axis=1,keys=eret_names)
        vw_table=pd.concat(s_ws,axis=1,keys=eret_names)
        return eq_table,vw_table

    def univariate_portfolio_analysis_with_k_month_ahead_returns(self):
        eq_tables=[]
        vw_tables=[]
        for indicator in self.indicators:
            eq_table,vw_table=self._one_indicator(indicator)
            eq_tables.append(eq_table)
            vw_tables.append(vw_table)
            print(indicator)

        eq=pd.concat(eq_tables,axis=0,keys=self.indicators)
        vw=pd.concat(vw_tables,axis=0,keys=self.indicators)

        eq.to_csv(os.path.join(self.path,'univariate portfolio analysis_k-month-ahead-returns-eq.csv'))
        vw.to_csv(os.path.join(self.path,'univariate portfolio analysis_k-month-ahead-returns-vw.csv'))

    def run(self):
        self.summary()
        self.correlation()
        self.persistence()
        self.breakPoints_and_countGroups()
        self.portfolio_analysis_with_ff3_alpha()
        self.univariate_portfolio_analysis_with_k_month_ahead_returns()
        self.fm()


class Bivariate(CrossSection):
    q=5

    def __init__(self,indicators,proj_path):
        '''
        :param factornames:list the name of the factors to analyse,such as ['size','beta']
        :param proj_path:
        '''
        super().__init__()
        self.indicators=indicators
        self.proj_path=proj_path

    def _get_independent_data(self):
        # TODO: add the method of ratios such as [0.3,0.7]
        # sometimes the self.indicators and ['mktCap','eret'] may share some elements
        comb=dataset.get_by_indicators(list(set(self.indicators+['mktCap','eret'])))
        comb=comb.dropna()
        comb['g1']=comb.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[self.indicators[0]],self.q,
                              labels=[self.indicators[0]+str(i) for i in range(1,self.q+1)])
        )

        comb['g2']=comb.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[self.indicators[1]],self.q,
                              labels=[self.indicators[1]+str(i) for i in range(1,self.q+1)])
        )
        return comb


    def _get_dependent_data(self,indicators):
        '''

        :param indicators:list with two elements,the first is the controlling variable
        :return:
        '''

        # sometimes the self.indicators and ['mktCap','eret'] may share some elements
        comb=dataset.get_by_indicators(list(set(self.indicators+['mktCap','eret'])))
        comb=comb.dropna()
        comb['g1']=comb.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[indicators[0]],self.q,
                              labels=[indicators[0]+str(i) for i in range(1,self.q+1)])
        )

        comb['g2']=comb.groupby(['t','g1'],group_keys=False).apply(
            lambda df:pd.qcut(df[indicators[1]],self.q,
                              labels=[indicators[1]+str(i) for i in range(1,self.q+1)])
        )

        return comb

    def _get_eret(self,comb):
        group_eavg_ts = comb.groupby(['g1', 'g2', 't'])['eret'].mean()
        group_wavg_ts = comb.groupby(['g1', 'g2', 't']).apply(
            lambda df: np.average(df['eret'], weights=df['mktCap']))
        return group_eavg_ts,group_wavg_ts

    def _dependent_portfolio_analysis(self, group_ts, controlGroup='g1', targetGroup='g2'):
        # Table 9.6
        controlIndicator= group_ts.index.get_level_values(controlGroup)[0][:-1]
        targetName= group_ts.index.get_level_values(targetGroup)[0][:-1]

        #A
        a_data=group_ts.groupby(['t', controlGroup, targetGroup]).mean().unstack(level=[controlGroup])
        a_data.columns=a_data.columns.astype(str)

        #A1
        a1_data=group_ts.groupby(['t', controlGroup, targetGroup]).mean().groupby(['t', targetGroup]).mean()
        a_data[controlIndicator+' avg']=a1_data
        _a=a_data.groupby(targetGroup).mean()

        def _get_spread(df):
            time=df.index.get_level_values('t')[0]
            return df.loc[(time,targetName+str(self.q))]-df.loc[(time,targetName+'1')]
        #B
        b_data=a_data.groupby('t').apply(_get_spread)
        _b1=b_data.apply(self._excess_ret)
        _b2=b_data.apply(self._alpha)

        _b1.index=[targetName+str(self.q)+'-1',targetName+str(self.q)+'-1 t']
        _b2.index=[targetName+str(self.q)+'-1 capm alpha',targetName+str(self.q)+'-1 capm alpha t']

        _a.index=_a.index.astype(str)
        _a.columns=_a.columns.astype(str)

        return pd.concat([_a,_b1,_b2],axis=0)


    def _dependent_portfolio_analysis0(self,group_ts,col='g1',ind='g2'):
        # Table 9.6
        colname=group_ts.index.get_level_values(col)[0][:-1]
        indname=group_ts.index.get_level_values(ind)[0][:-1]

        #A
        a_data=group_ts.groupby(['t',col,ind]).mean().unstack(level=[col])
        a_data.columns=a_data.columns.astype(str)

        #A1
        a1_data=group_ts.groupby(['t',col,ind]).mean().groupby(['t',ind]).mean()
        a_data[colname+' avg']=a1_data
        _a=a_data.groupby(ind).mean()

        def _get_spread(df):
            time=df.index.get_level_values('t')[0]
            return df.loc[(time,indname+str(self.q))]-df.loc[(time,indname+'1')]
        #B
        b_data=a_data.groupby('t').apply(_get_spread)
        _b1=b_data.apply(self._excess_ret)
        _b2=b_data.apply(self._alpha)

        _b1.index=[indname+str(self.q)+'-1',indname+str(self.q)+'-1 t']
        _b2.index=[indname+str(self.q)+'-1 capm alpha',indname+str(self.q)+'-1 capm alpha t']

        _a.index=_a.index.astype(str)
        _a.columns=_a.columns.astype(str)

        return pd.concat([_a,_b1,_b2],axis=0)

    def _dependent_portfolio_analysis_twin(self, group_ts,controlGroup='g2',targetGroup='g1'):
        #table 10.5 panel B
        group_ts.head()

        targetIndicator = group_ts.index.get_level_values(targetGroup)[0][:-1]# targetGroup
        controlIndicator = group_ts.index.get_level_values(controlGroup)[0][:-1]#controlGroup

        a1_data=group_ts.groupby(['t', targetGroup, controlGroup]).mean().groupby(['t', targetGroup]).mean()

        stk=a1_data.unstack()
        stk.index=stk.index.astype(str)
        stk.columns=stk.columns.astype(str)
        stk[targetIndicator+str(self.q)+'-1']=stk[targetIndicator+str(self.q)]-stk[targetIndicator+'1']
        _a=stk.apply(self._excess_ret)
        _b=stk.apply(self._alpha)

        table=pd.concat([_a,_b],axis=0)
        return table

    def _independent_portfolio_analysis(self,group_ts):
        #table 9.8

        table1=self._dependent_portfolio_analysis(group_ts, controlGroup='g1', targetGroup='g2')
        table2=self._dependent_portfolio_analysis(group_ts, controlGroup='g2', targetGroup='g1').T
        table1,table2=get_outer_frame([table1,table2])
        table=table1.fillna(table2)
        return table

    def independent_portfolio_analysis(self):
        comb=self._get_independent_data()
        group_eavg_ts, group_wavg_ts = self._get_eret(comb)

        table_eavg = self._independent_portfolio_analysis(group_eavg_ts)
        table_wavg = self._independent_portfolio_analysis(group_wavg_ts)
        table_eavg.to_csv(os.path.join(self.proj_path,
            'bivariate independent-sort portfolio analysis_equal weighted_%s_%s.csv'%(self.indicators[0],self.indicators[1])))
        table_wavg.to_csv(os.path.join(self.proj_path,
            'bivariate independent-sort portfolio analysis_value weighted_%s_%s.csv'%(self.indicators[0],self.indicators[1])))

    def dependent_portfolio_analysis(self):
        def _f(indicators):
            comb=self._get_dependent_data(indicators)
            group_eavg_ts, group_wavg_ts = self._get_eret(comb)

            table_eavg = self._dependent_portfolio_analysis(group_eavg_ts)
            table_wavg = self._dependent_portfolio_analysis(group_wavg_ts)
            table_eavg.to_csv(os.path.join(self.proj_path,
                'bivariate dependent-sort portfolio analysis_equal weighted_%s_%s.csv'%(indicators[0],indicators[1])))
            table_wavg.to_csv(os.path.join(self.proj_path,
                'bivariate dependent-sort portfolio analysis_value weighted_%s_%s.csv'%(indicators[0],indicators[1])))

        _f(self.indicators)
        _f(self.indicators[::-1])

    def dependent_portfolio_analysis_twin(self):
        def _f(indicators):
            comb = self._get_dependent_data(indicators)
            group_eavg_ts, group_wavg_ts = self._get_eret(comb)

            table_eavg = self._dependent_portfolio_analysis_twin(group_eavg_ts)
            table_wavg = self._dependent_portfolio_analysis_twin(group_wavg_ts)

            table_eavg.to_csv(os.path.join(self.proj_path,
                                           'bivariate dependent-sort portfolio analysis_twin_equal weighted_%s_%s.csv' % (
                                           indicators[0], indicators[1])))
            table_wavg.to_csv(os.path.join(self.proj_path,
                                           'bivariate dependent-sort portfolio analysis_twin_weighted_%s_%s.csv' % (
                                           indicators[0], indicators[1])))

        _f(self.indicators)
        _f(self.indicators[::-1])

    def _fm(self,ll_indeVars):
        '''
        :param ll_indeVars: list of list,the inside list contains all
            the indepedent variables to construct a regress equation

        :return:
        '''
        indicators=list(set(var for l_indeVars in ll_indeVars for var in l_indeVars))+['eret']
        comb=dataset.get_by_indicators(indicators)
        comb=comb.reset_index()

        stks=[]
        for l_indeVars in ll_indeVars:
            '''
            replace the olde name with new name,since patsy do not support name starts with number 
            
            '''
            newname=['name'+str(i) for i in range(1,len(l_indeVars)+1)]
            df=comb[l_indeVars+['t','eret']].dropna()
            df.columns=newname+['t','eret']
            formula='eret ~ '+' + '.join(newname)
            #TODO:lags?
            r,adj_r2,n=famaMacBeth(formula,'t',df,lags=5)
            r=r.rename(index=dict(zip(newname,l_indeVars)))
            stk=r[['coef','tvalue']].stack()
            stk.index=stk.index.map('{0[0]} {0[1]}'.format)
            stk['adj_r2']=adj_r2
            stk['n']=n
            stks.append(stk)

        table=pd.concat(stks,axis=1,keys=range(1,len(ll_indeVars)+1))

        newIndex=[var+' '+suffix for var in indicators for suffix in ['coef','tvalue']]+ \
            ['Intercept coef','Intercept tvalue','adj_r2','n']

        table=table.reindex(index=newIndex)

        table.to_csv(os.path.join(os.path.join(self.proj_path,'fama macbeth regression analysis.csv')))


class Size_beta_bivariate(Bivariate):
    def __init__(self):
        indicators=['size','12M']
        proj_path = r'D:\zht\database\quantDb\researchTopics\assetPricing1\tmp'
        super().__init__(indicators,proj_path)

    def fm(self):
        ll_indeVars = [['size'], ['size', '12M'], ['size_ff'], ['size_ff', '12M']]
        super()._fm(ll_indeVars)

    def run(self):
        self.dependent_portfolio_analysis()
        self.independent_portfolio_analysis()
        self.fm()

#----------chapter 10
class Value_beta_bivariate(Bivariate):
    def __init__(self):
        indicators=['bm','12M']
        proj_path=r'D:\zht\database\quantDb\researchTopics\assetPricing1\tmp1'
        super().__init__(indicators,proj_path)

    def run(self):
        self.dependent_portfolio_analysis()
        self.independent_portfolio_analysis()
        self.dependent_portfolio_analysis_twin()

class Value_size_bivariate(Bivariate):
    def __init__(self):
        indicators=['bm','mktCap']
        proj_path=r'D:\zht\database\quantDb\researchTopics\assetPricing1\tmp1'
        super().__init__(indicators,proj_path)

    def fm(self):
        ll_indeVars = [['bm'], ['bm', '12M'], ['bm','size'], ['bm','12M', 'size'],
                       ['logbm'],['logbm','12M'],['logbm','size'],['logbm','12M','size']]
        super()._fm(ll_indeVars)

    def run(self):
        self.dependent_portfolio_analysis()
        self.independent_portfolio_analysis()
        self.dependent_portfolio_analysis_twin()
        self.fm()

class Mom_bivariate(Bivariate):
    #TODO: table 11.6
    raise NotImplementedError


def chapter10():
    Value_beta_bivariate().run()
    Value_size_bivariate().run()

def chapter11():
    Univariate_mom().run()





if __name__=='__main__':
    # bivariate_beta = Beta()
    # bivariate_beta.run()
    #
    # bivariate_size = Size()
    # bivariate_size.run()
    #
    # value=Value()
    # value.run()
    #

    # Size_beta_bivariate().run()

    # chapter10()

    chapter11()




