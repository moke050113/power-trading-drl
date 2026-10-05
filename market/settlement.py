class Settlement:
    """
    简化的电力交易结算模型
    """

    def calculate_total_cost(self, long_term_cost, spot_cost):
        """
        计算总购电成本

        参数：
            long_term_cost：中长期购电成本，单位：元
            spot_cost：现货购电成本，单位：元

        返回：
            total_cost：总购电成本，单位：元
        """

        total_cost = long_term_cost + spot_cost

        return total_cost