def gene_to_num_str(gene_name, gene_type):
    """Strips excess gene name info to number string.
    
    Parameters
    ----------
    gene_name : str
        Gene or allele name
    gene_type : char
        Genomic cassette type. (i.e. V, D, or J)
    Returns
    -------
    num_str : str
        Reduced gene or allele name with leading zeros and excess
        characters removed.
        
    """

    num_str = gene_name.lower().split(gene_type.lower())[-1]
    num_str = '-'.join([g.lstrip('0') for g in num_str.split('-')])
    num_str = '*'.join([g.lstrip('0') for g in num_str.split('*')])
    
    return gene_type.lower()+num_str